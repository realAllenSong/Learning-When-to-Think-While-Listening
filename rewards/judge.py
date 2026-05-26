"""
Async batched LLM judge for R_t and R_c.
"""

import concurrent.futures
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class JudgeRequest:
    episode_id: str
    prompt: str
    callback: Optional[Callable[[str, float], None]] = None


def _parse_score(output: str) -> float:
    import re

    text = output.strip().lower()

    # First, try the most common failure case for reasoning-capable judge models:
    # they may emit extra analysis (or even `<think>...</think>`) and then end
    # with the actual numeric score on the last line.
    allowed_matches = re.findall(r"(?<![0-9])(1(?:\.0)?|0\.5|0(?:\.0)?)(?![0-9])", text)
    if allowed_matches:
        value = allowed_matches[-1]
        if value in {"1", "1.0"}:
            return 1.0
        if value == "0.5":
            return 0.5
        if value in {"0", "0.0"}:
            return 0.0

    match = re.search(r"(?:score|rating|quality)[\s:=]+([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+)", text)
    if match:
        num, denom = float(match.group(1)), float(match.group(2))
        if denom > 0:
            return min(1.0, max(0.0, num / denom))

    match = re.search(r"^([0-9]+(?:\.[0-9]+)?)$", text.strip())
    if match:
        value = float(match.group(1))
        return min(1.0, max(0.0, value if value <= 1.0 else value / 10.0))

    positive = {"yes", "consistent", "correct", "good", "accurate", "valid"}
    negative = {"no", "inconsistent", "incorrect", "poor", "inaccurate", "invalid"}
    if any(word in text for word in positive):
        return 1.0
    if any(word in text for word in negative):
        return 0.0
    return 0.5


def _resolve_completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/completions"
    return endpoint + "/v1/completions"


def _resolve_chat_completion_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


class _JudgeBackend:
    def score_prompts(self, prompts: List[str]) -> List[float]:
        raise NotImplementedError


def _is_context_length_http_error(code: int, body: str) -> bool:
    text = str(body or "").lower()
    return (
        int(code) == 400
        and (
            "maximum context length" in text
            or "input_tokens" in text
            or "reduce the length of the input prompt" in text
        )
    )


class _TransformersJudgeBackend(_JudgeBackend):
    def __init__(self, model_name: str, max_new_tokens: int, device: str):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._torch = None
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    def score_prompts(self, prompts: List[str]) -> List[float]:
        self._load()
        if not prompts:
            return []

        with self._torch.no_grad():
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._model.device)
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        results = []
        for out_ids, prompt_len in zip(output_ids, prompt_lens):
            gen_text = self._tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)
            results.append(_parse_score(gen_text))
        return results


class _HTTPJudgeBackend(_JudgeBackend):
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: str,
        max_new_tokens: int,
        timeout_sec: float,
        concurrency: int = 1,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec
        self.concurrency = max(1, int(concurrency or 1))

    @staticmethod
    def _message_text(choice: Dict[str, Any]) -> str:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        if content is not None:
            return str(content)
        return str(choice.get("text", ""))

    def _score_prompt(self, prompt: str) -> float:
        request = urllib.request.Request(
            _resolve_chat_completion_url(self.endpoint),
            data=json.dumps(
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": self.max_new_tokens,
                    "include_reasoning": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stop": ["\n"],
                    "n": 1,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.api_key),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - live-only path
            body = exc.read().decode("utf-8", errors="replace")
            if _is_context_length_http_error(exc.code, body):
                return 0.0
            raise RuntimeError("judge HTTP {}: {}".format(exc.code, body)) from exc

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("judge returned 0 choices for prompt")
        return _parse_score(self._message_text(dict(choices[0])))

    def score_prompts(self, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        if self.concurrency <= 1 or len(prompts) <= 1:
            return [self._score_prompt(prompt) for prompt in prompts]
        max_workers = min(self.concurrency, len(prompts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._score_prompt, prompts))


class LLMJudge:
    _instances: Dict[str, "LLMJudge"] = {}

    def __init__(
        self,
        model_name: str = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen3.5-27B"),
        batch_size: int = 16,
        max_new_tokens: int = 32,
        device: str = os.environ.get("LLM_JUDGE_DEVICE_MAP", "auto"),
        endpoint: str = os.environ.get("LLM_JUDGE_ENDPOINT", ""),
        api_key: str = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY"),
        http_concurrency: int = 1,
        timeout_sec: float = 120.0,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.endpoint = endpoint
        self.api_key = api_key
        self.http_concurrency = max(1, int(http_concurrency or 1))
        self.timeout_sec = timeout_sec

        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._results: Dict[str, float] = {}
        self._lock = threading.Lock()

        if self.endpoint:
            self._backend: _JudgeBackend = _HTTPJudgeBackend(
                endpoint=self.endpoint,
                model_name=self.model_name,
                api_key=self.api_key,
                max_new_tokens=self.max_new_tokens,
                timeout_sec=self.timeout_sec,
                concurrency=self.http_concurrency,
            )
        else:
            self._backend = _TransformersJudgeBackend(
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                device=self.device,
            )

    @classmethod
    def get_instance(
        cls,
        model_name: str = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen3.5-27B"),
        **kwargs,
    ) -> "LLMJudge":
        cache_key = "{}::{}::{}::{}".format(
            model_name,
            kwargs.get("endpoint", os.environ.get("LLM_JUDGE_ENDPOINT", "")),
            kwargs.get("batch_size", 16),
            kwargs.get("http_concurrency", 1),
        )
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(model_name=model_name, **kwargs)
        return cls._instances[cache_key]

    def _run_batch(self, requests: List[JudgeRequest]) -> None:
        scores = self._backend.score_prompts([request.prompt for request in requests])
        for request, score in zip(requests, scores):
            with self._lock:
                self._results[request.episode_id] = score
            if request.callback is not None:
                request.callback(request.episode_id, score)

    def _worker_loop(self) -> None:
        buffer: List[JudgeRequest] = []
        while self._running:
            try:
                buffer.append(self._queue.get(timeout=0.05))
                if len(buffer) >= self.batch_size:
                    self._run_batch(buffer)
                    buffer.clear()
            except queue.Empty:
                if buffer:
                    self._run_batch(buffer)
                    buffer.clear()

        if buffer:
            self._run_batch(buffer)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="judge-worker")
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=10)

    def submit(
        self,
        episode_id: str,
        prompt: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._queue.put(JudgeRequest(episode_id=episode_id, prompt=prompt, callback=callback))

    def flush(self) -> None:
        pending: List[JudgeRequest] = []
        while not self._queue.empty():
            try:
                pending.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if pending:
            self._run_batch(pending)

    def get_result(self, episode_id: str) -> Optional[float]:
        with self._lock:
            return self._results.get(episode_id)

    def score_batch(self, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        if isinstance(self._backend, _HTTPJudgeBackend):
            return self._backend.score_prompts(prompts)
        batch_size = max(1, int(self.batch_size or 1))
        scores: List[float] = []
        for start in range(0, len(prompts), batch_size):
            scores.extend(self._backend.score_prompts(prompts[start : start + batch_size]))
        return scores

    def score_sync(self, prompt: str) -> float:
        return self.score_batch([prompt])[0]

    def __repr__(self) -> str:
        mode = "http" if self.endpoint else "transformers"
        return f"LLMJudge(model={self.model_name!r}, mode={mode}, batch_size={self.batch_size})"
