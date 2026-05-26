"""
Local vLLM service controller for online rollout reloads.
"""

from __future__ import annotations

import os
import json
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LocalVLLMServiceConfig:
    python_bin: str
    host: str = "127.0.0.1"
    port: int = 8100
    served_model_name: str = "Qwen/Qwen2.5-Omni-7B"
    cuda_visible_devices: str = "0"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    ready_timeout_sec: float = 600.0
    enable_lora: bool = False
    max_loras: int = 1
    lora_name: str = ""
    lora_base_model_name: str = ""
    reload_max_attempts: int = 2
    reload_retry_sleep_sec: float = 15.0


class LocalVLLMServiceController:
    def __init__(
        self,
        config: LocalVLLMServiceConfig,
        initial_model: str,
        initial_lora_path: str = "",
        extra_args: Optional[List[str]] = None,
    ):
        self.config = config
        self.initial_model = initial_model
        self.initial_lora_path = str(initial_lora_path or "").strip()
        self.extra_args = list(extra_args or [])
        self._process: Optional[subprocess.Popen] = None
        self._current_model = ""
        self._current_lora_path = ""

    @property
    def current_model(self) -> str:
        return self._current_model or self.initial_model

    @property
    def ready_url(self) -> str:
        return "http://{}:{}/v1/models".format(self.config.host, self.config.port)

    def _resolve_lora_dir(self, path_str: str) -> str:
        raw = str(path_str or "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            if (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists():
                return str(path)
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}
                lora_metadata = metadata.get("lora") if isinstance(metadata, dict) else {}
                if isinstance(lora_metadata, dict):
                    adapter_subdir = str(lora_metadata.get("adapter_subdir") or "").strip()
                    if adapter_subdir:
                        adapter_dir = path / adapter_subdir
                        if (adapter_dir / "adapter_config.json").exists() and (
                            adapter_dir / "adapter_model.safetensors"
                        ).exists():
                            return str(adapter_dir)
        return str(path)

    def _wait_for_ready(self) -> None:
        deadline = time.time() + float(self.config.ready_timeout_sec)
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "local vLLM service exited early with code {}".format(self._process.returncode)
                )
            try:
                with urllib.request.urlopen(self.ready_url, timeout=2.0) as response:
                    if response.status < 500:
                        return
            except Exception:
                time.sleep(2.0)
        raise RuntimeError("timed out waiting for local vLLM service at {}".format(self.ready_url))

    def _launch(self, model_name_or_path: str, lora_path: str = "") -> None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.cuda_visible_devices)
        env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
        env["PYTHONUNBUFFERED"] = "1"
        python_bin_dir = os.path.abspath(os.path.dirname(str(self.config.python_bin)))
        env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")

        command = [
            self.config.python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name_or_path,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--dtype",
            self.config.dtype,
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--max-model-len",
            str(self.config.max_model_len),
            "--served-model-name",
            self.config.served_model_name,
        ]
        if self.config.trust_remote_code:
            command.append("--trust-remote-code")
        if int(self.config.tensor_parallel_size or 1) > 1:
            command.extend(["--tensor-parallel-size", str(int(self.config.tensor_parallel_size))])
        resolved_lora_dir = self._resolve_lora_dir(lora_path)
        if self.config.enable_lora and resolved_lora_dir:
            lora_payload = {
                "name": str(self.config.lora_name or "wait-think-answer-lora"),
                "path": resolved_lora_dir,
            }
            if str(self.config.lora_base_model_name or "").strip():
                lora_payload["base_model_name"] = str(self.config.lora_base_model_name).strip()
            command.extend(
                [
                    "--enable-lora",
                    "--max-loras",
                    str(int(self.config.max_loras or 1)),
                    "--lora-modules",
                    json.dumps(lora_payload, ensure_ascii=False),
                ]
            )
        command.extend(self.extra_args)

        self._process = subprocess.Popen(command, env=env, start_new_session=True)
        try:
            self._wait_for_ready()
        except Exception:
            self.stop()
            raise
        self._current_model = str(self.config.lora_name or model_name_or_path)
        self._current_lora_path = resolved_lora_dir

    def start(self) -> Dict[str, str]:
        if self._process is not None and self._process.poll() is None:
            return {"status": "already-running", "model": self.current_model, "ready_url": self.ready_url}
        self._launch(self.initial_model, self.initial_lora_path)
        return {"status": "started", "model": self.current_model, "ready_url": self.ready_url}

    def stop(self) -> Dict[str, str]:
        if self._process is None:
            return {"status": "not-running"}
        if self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception:
                self._process.terminate()
            try:
                self._process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except Exception:
                    self._process.kill()
                self._process.wait(timeout=30.0)
        code = self._process.returncode
        self._process = None
        return {"status": "stopped", "returncode": str(code)}

    def reload(self, model_name_or_path: str) -> Dict[str, str]:
        stopped = self.stop()
        attempts = max(1, int(self.config.reload_max_attempts or 1))
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                if self.config.enable_lora:
                    self._launch(self.initial_model, model_name_or_path)
                else:
                    self._launch(model_name_or_path)
                return {
                    "status": "reloaded",
                    "model": self.current_model,
                    "ready_url": self.ready_url,
                    "previous_status": stopped.get("status", "unknown"),
                    "reload_attempt": str(attempt),
                }
            except Exception as exc:
                last_error = exc
                self.stop()
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.reload_retry_sleep_sec or 0.0)))
        raise RuntimeError(
            "failed to reload local vLLM service after {} attempt(s): {}".format(attempts, last_error)
        ) from last_error
