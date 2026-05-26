"""
Trainable Omni actor updater for DAPO controller training.

The rollout policy is served by an OpenAI-compatible endpoint. The local actor
scores sampled trajectories turn by turn and applies the configured GRPO/DAPO
objective with optional KL/reference-model support.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .checkpointing import CheckpointArtifact
from .dapo_loss import (
    approximate_observed_kl_from_token_logprobs,
    clip_fraction_from_token_logprobs,
    clipped_dapo_token_objective,
    overlong_shaping_penalty,
)
from .grpo_loss import (
    approximate_reverse_kl,
    approximate_observed_kl_from_sequence_logprobs,
    clip_fraction_from_sequence_logprobs,
    clipped_grpo_objective,
    normalized_sequence_logprob_from_token_logprobs,
)
from .omni_actor import (
    build_turn_training_examples,
    prepare_turn_model_inputs_batch,
)
from rewards.reference_answer_fallback import (
    REFERENCE_ANSWER_JUDGE_SYSTEM_PROMPT,
    build_reference_answer_judge_prompt,
    build_reference_answer_judge_retry_prompt,
    has_reference_answer_judge_json,
    resolve_reference_answer_judge_output,
)


OMNI_REQUIRED_CHECKPOINT_FILES = ("spk_dict.pt",)
LOGGER = logging.getLogger(__name__)


@dataclass
class OmniActorUpdateConfig:
    model_name: str = "Qwen/Qwen2.5-Omni-7B"
    processor_name: str = ""
    reference_model_name: str = ""
    init_adapter_path: str = ""
    reference_adapter_path: str = ""
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"
    lora_adapter_dirname: str = "lora_adapter"
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    optimizer_name: str = "adamw"
    device_map: str = "auto"
    reference_device_map: str = ""
    model_max_memory: str = ""
    reference_max_memory: str = ""
    dtype: str = "bfloat16"
    grad_clip_norm: float = 1.0
    sampling_rate: int = 16000
    max_turn_examples: int = 0
    turn_batch_size: int = 1
    kl_beta: float = 0.005
    gradient_checkpointing: bool = False
    cache_audio_arrays: bool = True
    sequence_level_credit: bool = True
    update_epochs: int = 1
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    loss_mode: str = "grpo-sequence"
    overlong_shaping: bool = True
    overlong_threshold_tokens: int = 32
    overlong_penalty_slope: float = 0.03
    overlong_penalty_cap: float = 1.0
    warmup_steps: int = 50
    credit_assignment: str = "hybrid-local"
    hybrid_alpha: float = 0.5
    resume_checkpoint: str = ""
    checkpoint_mode: str = "full"
    load_optimizer_state: bool = True
    deepspeed_enabled: bool = False
    deepspeed_zero_stage: int = 0
    deepspeed_micro_batch_size: int = 0
    deepspeed_gradient_accumulation_steps: int = 1
    deepspeed_config_json: str = ""
    deepspeed_offload_optimizer_device: str = ""
    deepspeed_offload_param_device: str = ""
    answer_fallback_enabled: bool = False
    answer_fallback_max_new_tokens: int = 96


class TrainableOmniActorUpdater:
    def __init__(
        self,
        config: Optional[OmniActorUpdateConfig] = None,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        reference_model: Optional[Any] = None,
    ):
        self.config = config or OmniActorUpdateConfig()
        self._model = model
        self._processor = processor
        self._optimizer = optimizer
        self._reference_model = reference_model
        self._torch = None
        self._resume_metadata: Optional[Dict[str, Any]] = None
        self._using_lora = bool(self.config.use_lora)
        self._base_model_source = str(self.config.model_name or "")
        self._optimizer_kind = str(self.config.optimizer_name or "adamw").lower()
        self._optimizer_summary: Dict[str, Any] = {}
        self._current_lr_scale: float = 1.0
        self._actor_lora_attachment_scope: str = "thinker"
        self._reference_lora_attachment_scope: str = "thinker"
        self._deepspeed = None
        self._deepspeed_engine = None
        self._deepspeed_checkpoint_tag: str = ""

    def _resolve_pretrained_source(self, *candidates: Any) -> str:
        for raw_value in candidates:
            text = str(raw_value or "").strip()
            if not text:
                continue
            if text.startswith(("~", "/", "./", "../")):
                path = Path(text).expanduser()
                if path.exists():
                    return str(path)
                continue
            return text
        return ""

    def _trainable_parameters(self):
        if self._model is None:
            return []
        target_model = self._trainable_module()
        return [param for param in target_model.parameters() if param.requires_grad]

    def _trainable_named_parameters(self):
        if self._model is None:
            return []
        target_model = self._trainable_module()
        return [(name, param) for name, param in target_model.named_parameters() if param.requires_grad]

    def _trainable_module(self):
        if self._model is None:
            return None
        if self._using_lora and self._actor_lora_attachment_scope == "full-model":
            return self._model
        return getattr(self._model, "thinker", self._model)

    def _distributed_rank(self) -> int:
        torch_module = self._torch
        if torch_module is None:
            import torch as torch_module

        distributed = getattr(torch_module, "distributed", None)
        if distributed is None:
            return 0
        try:
            if not distributed.is_available() or not distributed.is_initialized():
                return 0
            return int(distributed.get_rank())
        except Exception:
            return 0

    def _is_main_process(self) -> bool:
        return self._distributed_rank() == 0

    def _gather_trainable_state_dict_for_save(self, model) -> Optional[Dict[str, Any]]:
        if model is None:
            return None
        named_parameters = [
            (name, param)
            for name, param in model.named_parameters()
            if getattr(param, "requires_grad", False)
        ]
        if not named_parameters:
            return None

        gather_context = nullcontext()
        if self._deepspeed_engine is not None and int(self.config.deepspeed_zero_stage or 0) >= 3:
            params_to_gather = [param for _, param in named_parameters if hasattr(param, "ds_id")]
            if params_to_gather:
                deepspeed_module = self._load_deepspeed_module()
                gather_context = deepspeed_module.zero.GatheredParameters(
                    params_to_gather,
                    modifier_rank=None,
                )

        with gather_context:
            return {
                name: param.detach().cpu().clone()
                for name, param in named_parameters
            }

    def _resolve_torch_dtype(self, torch_module):
        mapping = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        return mapping.get(str(self.config.dtype).lower(), torch_module.bfloat16)

    def _deepspeed_requested(self) -> bool:
        return bool(self.config.deepspeed_enabled) and int(self.config.deepspeed_zero_stage or 0) >= 0

    def _load_deepspeed_module(self):
        if self._deepspeed is None:
            import deepspeed

            self._deepspeed = deepspeed
        return self._deepspeed

    def _deepspeed_world_size(self) -> int:
        for key in ("WORLD_SIZE", "SLURM_NTASKS"):
            raw_value = str(os.environ.get(key, "")).strip()
            if not raw_value:
                continue
            try:
                return max(1, int(raw_value))
            except ValueError:
                continue
        return 1

    def _deepspeed_device_rank(self, torch_module) -> int:
        for param in self._trainable_parameters():
            device = getattr(param, "device", None)
            if device is None:
                continue
            if getattr(device, "type", "") == "cuda":
                return int(device.index or 0)
        if torch_module.cuda.is_available():
            return int(torch_module.cuda.current_device())
        return -1

    def _maybe_disable_optional_deepspeed_ops(self, deepspeed_module, *, world_size: int) -> None:
        if world_size > 1:
            return
        ops_module = getattr(deepspeed_module, "ops", None)
        compatible_ops = getattr(ops_module, "__compatible_ops__", None)
        if isinstance(compatible_ops, dict) and "deepspeed_shm_comm" in compatible_ops:
            # Single-process ZeRO runs do not need the optional shared-memory
            # collective helper, and disabling it avoids a runtime ninja/JIT
            # dependency for smoke jobs.
            compatible_ops["deepspeed_shm_comm"] = False

    def _deepspeed_local_rank(self, *, world_size: int, device_rank: int) -> int:
        if world_size <= 1:
            return device_rank if device_rank >= 0 else 0

        for key in ("LOCAL_RANK", "SLURM_LOCALID"):
            raw_value = str(os.environ.get(key, "")).strip()
            if not raw_value:
                continue
            try:
                return int(raw_value)
            except ValueError:
                continue
        return device_rank if device_rank >= 0 else 0

    def _ensure_deepspeed_distributed_backend(self, torch_module, deepspeed_module, *, device_rank: int) -> None:
        world_size = self._deepspeed_world_size()
        self._maybe_disable_optional_deepspeed_ops(deepspeed_module, world_size=world_size)

        comm_module = getattr(deepspeed_module, "comm", None)
        if comm_module is not None:
            try:
                if comm_module.is_initialized():
                    return
            except Exception:
                pass

        rank_text = str(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0").strip()
        try:
            rank = int(rank_text)
        except ValueError:
            rank = 0

        local_rank = self._deepspeed_local_rank(world_size=world_size, device_rank=device_rank)

        if world_size <= 1:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(local_rank)
        else:
            os.environ.setdefault("RANK", str(rank))
            os.environ.setdefault("WORLD_SIZE", str(world_size))
            os.environ.setdefault("LOCAL_RANK", str(local_rank))

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            seed_text = str(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_STEP_ID") or os.getpid()).strip()
            try:
                seed_value = int(seed_text)
            except ValueError:
                seed_value = os.getpid()
            os.environ["MASTER_PORT"] = str(20000 + (seed_value % 20000))

        if device_rank >= 0 and torch_module.cuda.is_available():
            torch_module.cuda.set_device(device_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        deepspeed_module.init_distributed(
            dist_backend=backend,
            auto_mpi_discovery=False,
            init_method="tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]),
            rank=rank,
            world_size=world_size,
            dist_init_required=True,
        )

    def _deepspeed_checkpoint_root(self, checkpoint_dir: Path) -> Path:
        return checkpoint_dir / "deepspeed"

    def _deepspeed_tag_for_step(self, step: int) -> str:
        return "step_{:06d}".format(int(step))

    def _deepspeed_config_dict(self) -> Dict[str, Any]:
        config_path = str(self.config.deepspeed_config_json or "").strip()
        if config_path:
            return json.loads(Path(config_path).read_text(encoding="utf-8"))

        micro_batch_size = max(
            1,
            int(
                self.config.deepspeed_micro_batch_size
                or self.config.turn_batch_size
                or 1
            ),
        )
        grad_accum = max(1, int(self.config.deepspeed_gradient_accumulation_steps or 1))
        dtype_key = str(self.config.dtype or "bfloat16").lower()
        zero_stage = max(0, int(self.config.deepspeed_zero_stage or 0))

        zero_optimization: Dict[str, Any] = {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": False,
            "reduce_scatter": bool(zero_stage >= 2),
        }
        offload_optimizer_device = str(self.config.deepspeed_offload_optimizer_device or "").strip().lower()
        if offload_optimizer_device:
            zero_optimization["offload_optimizer"] = {"device": offload_optimizer_device}
        offload_param_device = str(self.config.deepspeed_offload_param_device or "").strip().lower()
        if offload_param_device:
            zero_optimization["offload_param"] = {"device": offload_param_device}

        config: Dict[str, Any] = {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "zero_optimization": zero_optimization,
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,
        }
        if float(self.config.grad_clip_norm or 0.0) > 0:
            config["gradient_clipping"] = float(self.config.grad_clip_norm)

        if dtype_key in {"bfloat16", "bf16"}:
            config["bf16"] = {"enabled": True}
        elif dtype_key in {"float16", "fp16"}:
            config["fp16"] = {"enabled": True}
        else:
            config["bf16"] = {"enabled": False}
            config["fp16"] = {"enabled": False}
        return config

    def _parse_device_map(self, raw_value):
        if not isinstance(raw_value, str):
            return raw_value
        text = raw_value.strip().strip("'\"")
        if not text:
            return None
        cleaned = text.rstrip("}")
        if cleaned.isdigit():
            return {"": int(cleaned)}
        if cleaned.startswith("cuda:") and cleaned[5:].isdigit():
            return {"": int(cleaned[5:])}
        if text.isdigit():
            return {"": int(text)}
        if text.startswith("cuda:") and text[5:].isdigit():
            return {"": int(text[5:])}
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)
        return text

    def _normalize_max_memory_key(self, raw_key):
        key = str(raw_key).strip()
        if key.isdigit():
            return int(key)
        if key.startswith("cuda:") and key[5:].isdigit():
            return int(key[5:])
        return key

    def _parse_max_memory(self, raw_value: str) -> Optional[Dict[Any, str]]:
        text = str(raw_value or "").strip()
        if not text:
            return None
        if text.startswith("{"):
            parsed = json.loads(text)
            return {
                self._normalize_max_memory_key(key): str(value).strip()
                for key, value in parsed.items()
                if str(value).strip()
            }

        result: Dict[Any, str] = {}
        for part in [item.strip() for item in text.split(",") if item.strip()]:
            if "=" in part:
                key, value = part.split("=", 1)
            else:
                key, value = part.rsplit(":", 1)
            value = str(value).strip()
            if not value:
                continue
            result[self._normalize_max_memory_key(key)] = value
        return result or None

    def _device_map_spans_multiple_cuda_devices(self, device_map) -> bool:
        if device_map is None:
            return False
        if isinstance(device_map, str):
            text = device_map.strip().lower()
            if text == "auto":
                return True
            if text.startswith("cuda:") and text[5:].isdigit():
                return False
            return False

        device_indices = set()
        values = []
        if isinstance(device_map, dict):
            values = list(device_map.values())
        elif isinstance(device_map, (list, tuple)):
            values = list(device_map)

        for raw_value in values:
            if isinstance(raw_value, int):
                device_indices.add(int(raw_value))
                continue
            text = str(raw_value).strip().lower()
            if text.startswith("cuda:") and text[5:].isdigit():
                device_indices.add(int(text[5:]))
                continue
            if text.isdigit():
                device_indices.add(int(text))
                continue
            if text == "cpu":
                continue
            if text == "auto":
                return True

        return len(device_indices) > 1

    def _maybe_patch_qwen25_omni_rotary_embedding(self) -> None:
        torch_module = self._torch
        if torch_module is None:
            return
        try:
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        except Exception:
            return

        rotary_cls = getattr(modeling_module, "Qwen2_5OmniRotaryEmbedding", None)
        maybe_autocast = getattr(modeling_module, "maybe_autocast", None)
        if rotary_cls is None or maybe_autocast is None:
            return
        if getattr(rotary_cls, "_wta_multi_device_patch", False):
            return

        original_forward = rotary_cls.forward

        def patched_forward(rotary_self, x, position_ids):
            target_device = getattr(x, "device", None)
            inv_freq = rotary_self.inv_freq[None, None, :, None]
            position_values = position_ids[:, :, None, :]
            if target_device is not None:
                inv_freq_expanded = inv_freq.to(device=target_device, dtype=torch_module.float32).expand(
                    3,
                    position_ids.shape[1],
                    -1,
                    1,
                )
                position_ids_expanded = position_values.to(device=target_device, dtype=torch_module.float32)
            else:
                inv_freq_expanded = inv_freq.float().expand(3, position_ids.shape[1], -1, 1)
                position_ids_expanded = position_values.float()

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with maybe_autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
                emb = torch_module.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * rotary_self.attention_scaling
                sin = emb.sin() * rotary_self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        rotary_cls._wta_original_forward = original_forward
        rotary_cls.forward = patched_forward
        rotary_cls._wta_multi_device_patch = True
        LOGGER.info(
            "[omni-updater] patched Qwen2.5-Omni rotary embedding for multi-device actor/reference sharding"
        )

    def _build_model_kwargs(self, device_map, max_memory, torch_module) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(torch_module),
            "trust_remote_code": True,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        return kwargs

    def _apply_training_mode(self, model) -> None:
        model.train()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        target_model = getattr(model, "thinker", None)
        if target_model is not None and target_model is not model:
            target_model.train()
            if hasattr(target_model, "config"):
                target_model.config.use_cache = False
            if self.config.gradient_checkpointing and hasattr(target_model, "gradient_checkpointing_enable"):
                target_model.gradient_checkpointing_enable()

    def _build_optimizer(self, torch_module):
        optimizer_name = str(self.config.optimizer_name or "adamw").strip().lower()
        if optimizer_name != "adamw":
            raise ValueError("This public training release supports optimizer_name=adamw")
        named_parameters = self._trainable_named_parameters()
        self._optimizer_kind = "adamw"
        self._optimizer_summary = {
            "trainable_tensors": len(named_parameters),
            "trainable_parameters": sum(int(param.numel()) for _, param in named_parameters),
            "lr": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
        }
        return torch_module.optim.AdamW(
            [param for _, param in named_parameters],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _load_resume_metadata(self) -> Dict[str, Any]:
        if self._resume_metadata is not None:
            return self._resume_metadata
        resume_dir = str(self.config.resume_checkpoint or "").strip()
        if not resume_dir:
            self._resume_metadata = {}
            return self._resume_metadata
        metadata_path = Path(resume_dir) / "metadata.json"
        if not metadata_path.exists():
            self._resume_metadata = {}
            return self._resume_metadata
        self._resume_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._resume_metadata

    def _resume_lora_metadata(self) -> Dict[str, Any]:
        metadata = self._load_resume_metadata()
        payload = metadata.get("lora")
        if isinstance(payload, dict):
            return payload
        return {}

    def _resume_lora_adapter_dir(self) -> Optional[Path]:
        resume_dir = str(self.config.resume_checkpoint or "").strip()
        if not resume_dir:
            return None
        direct_adapter_dir = Path(resume_dir)
        if (direct_adapter_dir / "adapter_config.json").exists() and (
            direct_adapter_dir / "adapter_model.safetensors"
        ).exists():
            return direct_adapter_dir
        lora_metadata = self._resume_lora_metadata()
        adapter_subdir = str(
            lora_metadata.get("adapter_subdir")
            or self.config.lora_adapter_dirname
            or "lora_adapter"
        ).strip()
        if not adapter_subdir:
            return None
        adapter_dir = Path(resume_dir) / adapter_subdir
        if adapter_dir.exists():
            return adapter_dir
        return None

    def _lora_target_modules(self):
        raw_value = str(self.config.lora_target_modules or "").strip()
        if not raw_value:
            return "all-linear"
        lowered = raw_value.lower()
        if lowered in {"all-linear", "all_linear"}:
            return "all-linear"
        if raw_value.startswith("["):
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def _build_lora_config(self, peft_module):
        return peft_module.LoraConfig(
            r=int(self.config.lora_rank),
            lora_alpha=int(self.config.lora_alpha),
            lora_dropout=float(self.config.lora_dropout),
            bias="none",
            target_modules=self._lora_target_modules(),
            task_type=peft_module.TaskType.CAUSAL_LM,
        )

    def _read_adapter_target_spec(self, adapter_dir: Optional[Path]) -> Any:
        if adapter_dir is None:
            return None
        config_path = Path(adapter_dir) / "adapter_config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload.get("target_modules")

    def _target_spec_requires_full_model_scope(self, target_spec: Any) -> bool:
        if isinstance(target_spec, str):
            text = target_spec.strip()
            if not text:
                return False
            return "thinker." in text or "thinker\\." in text
        if isinstance(target_spec, (list, tuple)):
            for item in target_spec:
                text = str(item).strip()
                if "thinker." in text or "thinker\\." in text:
                    return True
        return False

    def _resolve_lora_attachment_scope(self, *, adapter_dir: Optional[Path] = None) -> str:
        target_spec = self._read_adapter_target_spec(adapter_dir)
        if target_spec is None:
            target_spec = self.config.lora_target_modules
        if self._target_spec_requires_full_model_scope(target_spec):
            return "full-model"
        return "thinker"

    def _lora_attach_target(self, model, *, scope: str):
        if scope == "full-model":
            return model
        return getattr(model, "thinker", model)

    def _attach_lora_adapter(self, model, *, adapter_dir: Optional[Path] = None, scope: str = "thinker"):
        from peft import PeftModel, get_peft_model

        target_model = self._lora_attach_target(model, scope=scope)
        if adapter_dir is not None and adapter_dir.exists():
            return PeftModel.from_pretrained(
                target_model,
                str(adapter_dir),
                is_trainable=True,
            )
        return get_peft_model(target_model, self._build_lora_config(__import__("peft")))

    def _install_lora_adapter(self, model, attached_model, *, scope: str):
        if scope == "full-model":
            return attached_model
        if hasattr(model, "thinker"):
            model.thinker = attached_model
            return model
        return attached_model

    def _save_lora_adapter(self, adapter_dir: Path) -> Path:
        target_model = self._trainable_module()
        adapter_dir.mkdir(parents=True, exist_ok=True)
        save_kwargs: Dict[str, Any] = {"is_main_process": self._is_main_process()}
        state_dict = self._gather_trainable_state_dict_for_save(target_model)
        if state_dict is not None:
            save_kwargs["state_dict"] = state_dict
        try:
            target_model.save_pretrained(str(adapter_dir), **save_kwargs)
        except TypeError as exc:
            unsupported_kwarg = "unexpected keyword argument" in str(exc)
            if not unsupported_kwarg:
                raise
            target_model.save_pretrained(str(adapter_dir))
        return adapter_dir

    def _export_merged_lora_model(self, checkpoint_dir: Path, adapter_dir: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5OmniForConditionalGeneration

        export_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self._base_model_source,
            torch_dtype=self._resolve_torch_dtype(torch),
            trust_remote_code=True,
            device_map="cpu",
        )
        scope = self._resolve_lora_attachment_scope(adapter_dir=adapter_dir)
        if scope == "full-model":
            export_wrapper = PeftModel.from_pretrained(
                export_model,
                str(adapter_dir),
                is_trainable=False,
            )
            merged_model = export_wrapper.merge_and_unload(progressbar=False)
            merged_model.save_pretrained(checkpoint_dir)
            return
        export_thinker = PeftModel.from_pretrained(
            export_model.thinker,
            str(adapter_dir),
            is_trainable=False,
        )
        export_model.thinker = export_thinker.merge_and_unload(progressbar=False)
        export_model.save_pretrained(checkpoint_dir)

    def _lora_metadata_payload(self, adapter_subdir: str = "") -> Dict[str, Any]:
        return {
            "enabled": bool(self._using_lora),
            "base_model_name": str(self._base_model_source or self.config.model_name or ""),
            "rank": int(self.config.lora_rank),
            "alpha": int(self.config.lora_alpha),
            "dropout": float(self.config.lora_dropout),
            "target_modules": self.config.lora_target_modules,
            "attachment_scope": str(self._actor_lora_attachment_scope or ""),
            "adapter_subdir": str(adapter_subdir or ""),
        }

    def _maybe_restore_optimizer(self) -> None:
        resume_dir = str(self.config.resume_checkpoint or "").strip()
        if (
            not resume_dir
            or self._optimizer is None
            or self._torch is None
            or not bool(self.config.load_optimizer_state)
        ):
            return
        optimizer_path = Path(resume_dir) / "optimizer.pt"
        if not optimizer_path.exists():
            return
        state_dict = self._torch.load(str(optimizer_path), map_location="cpu")
        self._optimizer.load_state_dict(state_dict)

    def _maybe_restore_deepspeed_engine(self) -> None:
        resume_dir = str(self.config.resume_checkpoint or "").strip()
        if (
            not resume_dir
            or self._deepspeed_engine is None
            or not self._deepspeed_requested()
        ):
            return
        metadata = self._load_resume_metadata()
        payload = metadata.get("deepspeed")
        if not isinstance(payload, dict):
            return
        ds_root = Path(resume_dir) / str(payload.get("checkpoint_subdir") or "deepspeed")
        tag = str(payload.get("tag") or "").strip()
        if not ds_root.exists() or not tag:
            return
        self._deepspeed_engine.load_checkpoint(
            str(ds_root),
            tag=tag,
            load_optimizer_states=bool(self.config.load_optimizer_state),
            load_lr_scheduler_states=False,
        )
        self._deepspeed_checkpoint_tag = tag

    def _initialize_deepspeed_engine(self, torch_module) -> None:
        deepspeed_module = self._load_deepspeed_module()
        device_rank = self._deepspeed_device_rank(torch_module)
        self._ensure_deepspeed_distributed_backend(
            torch_module,
            deepspeed_module,
            device_rank=device_rank,
        )
        optimizer = self._build_optimizer(torch_module)
        ds_config = self._deepspeed_config_dict()
        engine, optimizer, _, _ = deepspeed_module.initialize(
            args=SimpleNamespace(device_rank=device_rank),
            model=self._model,
            model_parameters=self._trainable_parameters(),
            optimizer=optimizer,
            config=ds_config,
            dist_init_required=False,
        )
        self._deepspeed_engine = engine
        self._model = engine.module
        self._optimizer = optimizer
        for param_group in self._optimizer.param_groups:
            param_group.setdefault(
                "initial_lr",
                float(param_group.get("lr", self.config.learning_rate)),
            )
        self._maybe_restore_deepspeed_engine()

    def _effective_init_adapter_dir(self) -> Optional[Path]:
        init_adapter = str(self.config.init_adapter_path or "").strip()
        if init_adapter:
            adapter_dir = Path(init_adapter)
            if adapter_dir.exists():
                return adapter_dir
        return self._resume_lora_adapter_dir()

    def _effective_reference_adapter_dir(self) -> Optional[Path]:
        ref_adapter = str(self.config.reference_adapter_path or "").strip()
        if ref_adapter:
            adapter_dir = Path(ref_adapter)
            if adapter_dir.exists():
                return adapter_dir
        return None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None and self._optimizer is not None:
            if self._torch is None:
                import torch

                self._torch = torch
            return

        import torch
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        self._torch = torch

        resume_lora = self._resume_lora_metadata()
        resume_adapter_dir = self._effective_init_adapter_dir()
        self._using_lora = (
            bool(self.config.use_lora)
            or bool(resume_lora.get("enabled"))
            or bool(resume_adapter_dir)
        )
        self._base_model_source = self._resolve_pretrained_source(
            resume_lora.get("base_model_name"),
            self.config.model_name,
            self.config.processor_name,
        )

        model_source = self._base_model_source if self._using_lora else self._resolve_pretrained_source(
            self.config.resume_checkpoint,
            self.config.model_name,
        )
        processor_name = self._resolve_pretrained_source(
            self.config.processor_name,
            self._base_model_source,
            self.config.model_name,
        )
        model_device_map = self._parse_device_map(self.config.device_map)
        model_max_memory = self._parse_max_memory(self.config.model_max_memory)
        if self._device_map_spans_multiple_cuda_devices(model_device_map):
            self._maybe_patch_qwen25_omni_rotary_embedding()

        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            processor_name,
            trust_remote_code=True,
        )
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_source,
            **self._build_model_kwargs(model_device_map, model_max_memory, torch),
        )

        if self._using_lora:
            self._actor_lora_attachment_scope = self._resolve_lora_attachment_scope(
                adapter_dir=resume_adapter_dir
            )
            attached_model = self._attach_lora_adapter(
                self._model,
                adapter_dir=resume_adapter_dir,
                scope=self._actor_lora_attachment_scope,
            )
            self._model = self._install_lora_adapter(
                self._model,
                attached_model,
                scope=self._actor_lora_attachment_scope,
            )

        self._apply_training_mode(self._model)
        if self._deepspeed_requested():
            self._initialize_deepspeed_engine(torch)
        else:
            self._optimizer = self._build_optimizer(torch)
            for param_group in self._optimizer.param_groups:
                param_group.setdefault("initial_lr", float(param_group.get("lr", self.config.learning_rate)))
            self._maybe_restore_optimizer()

        if (self.config.kl_beta > 0 or self.config.answer_fallback_enabled) and self._reference_model is None:
            reference_source = self._resolve_pretrained_source(
                self.config.reference_model_name,
                self._base_model_source,
                self.config.model_name,
            )
            reference_device_map = self._parse_device_map(
                self.config.reference_device_map or self.config.device_map
            )
            reference_max_memory = self._parse_max_memory(
                self.config.reference_max_memory or self.config.model_max_memory
            )
            if self._device_map_spans_multiple_cuda_devices(reference_device_map):
                self._maybe_patch_qwen25_omni_rotary_embedding()
            self._reference_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                reference_source,
                **self._build_model_kwargs(reference_device_map, reference_max_memory, torch),
            )
            reference_adapter_dir = self._effective_reference_adapter_dir()
            if reference_adapter_dir is not None:
                self._reference_lora_attachment_scope = self._resolve_lora_attachment_scope(
                    adapter_dir=reference_adapter_dir
                )
                attached_reference = self._attach_lora_adapter(
                    self._reference_model,
                    adapter_dir=reference_adapter_dir,
                    scope=self._reference_lora_attachment_scope,
                )
                self._reference_model = self._install_lora_adapter(
                    self._reference_model,
                    attached_reference,
                    scope=self._reference_lora_attachment_scope,
                )
            if getattr(self._reference_model, "has_talker", False):
                self._reference_model.disable_talker()
            self._reference_model.eval()
            if hasattr(self._reference_model, "config"):
                self._reference_model.config.use_cache = True
                self._reference_model.config.enable_audio_output = False
            for param in self._reference_model.parameters():
                param.requires_grad_(False)

    def _set_learning_rate_for_step(self, step_index: Optional[int]) -> float:
        if self._optimizer is None:
            return 1.0
        warmup_steps = max(0, int(self.config.warmup_steps or 0))
        if warmup_steps <= 0 or step_index is None or step_index <= 0:
            scale = 1.0
        else:
            scale = min(1.0, float(step_index) / float(warmup_steps))
        for param_group in self._optimizer.param_groups:
            initial_lr = float(param_group.get("initial_lr", param_group.get("lr", self.config.learning_rate)))
            param_group["lr"] = initial_lr * scale
        self._current_lr_scale = scale
        return scale

    def _checkpoint_file_sources(self) -> List[str]:
        candidates: List[str] = []
        for raw_value in (
            str(self.config.resume_checkpoint or "").strip(),
            str(getattr(self._model, "name_or_path", "") or "").strip(),
            str(self._base_model_source or "").strip(),
            str(self.config.model_name or "").strip(),
        ):
            if raw_value and raw_value not in candidates:
                candidates.append(raw_value)
        return candidates

    def _resolve_required_checkpoint_file(self, filename: str) -> Optional[str]:
        from transformers.utils import cached_file

        for source in self._checkpoint_file_sources():
            resolved = cached_file(
                source,
                filename,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if resolved:
                return str(resolved)
        return None

    def _copy_required_checkpoint_files(self, checkpoint_dir: Path) -> Dict[str, str]:
        copied: Dict[str, str] = {}
        for filename in OMNI_REQUIRED_CHECKPOINT_FILES:
            destination = checkpoint_dir / filename
            if destination.exists():
                copied[filename] = str(destination)
                continue

            resolved = self._resolve_required_checkpoint_file(filename)
            if not resolved:
                raise FileNotFoundError(
                    "Could not resolve required Omni checkpoint file '{}' from any source: {}".format(
                        filename,
                        ", ".join(self._checkpoint_file_sources()) or "<none>",
                    )
                )

            shutil.copy2(resolved, destination)
            copied[filename] = str(destination)
        return copied

    def _input_device_for_model(self, model=None):
        raw_model = model if model is not None else self._model
        candidates = []
        thinker_model = getattr(raw_model, "thinker", None)
        if thinker_model is not None and thinker_model is not raw_model:
            candidates.append(thinker_model)
        if raw_model is not None:
            candidates.append(raw_model)

        for target_model in candidates:
            get_input_embeddings = getattr(target_model, "get_input_embeddings", None)
            if callable(get_input_embeddings):
                try:
                    embeddings = get_input_embeddings()
                except NotImplementedError:
                    embeddings = None
                if embeddings is not None:
                    weight = getattr(embeddings, "weight", None)
                    device = getattr(weight, "device", None)
                    if device is not None:
                        return device

            device = getattr(target_model, "device", None)
            if device is not None:
                return device
            parameters = getattr(target_model, "parameters", None)
            if parameters is None:
                continue
            try:
                first_param = next(target_model.parameters())
            except StopIteration:
                continue
            return first_param.device
        return None

    def _forward_target_model(self, model=None):
        raw_model = model if model is not None else self._model
        thinker_model = getattr(raw_model, "thinker", None)
        if thinker_model is not None:
            return thinker_model
        return raw_model

    def _move_batch_to_device(self, payload: Dict[str, Any], model=None) -> Dict[str, Any]:
        torch = self._torch
        device = self._input_device_for_model(model)
        if torch is None or device is None:
            return payload

        moved = {}
        for key, value in payload.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _reference_answer_judge_tokenizer(self):
        processor = self._processor
        tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
        if tokenizer is None:
            raise RuntimeError("reference answer fallback judge requires a tokenizer on the Omni processor")
        tokenizer.padding_side = "left"
        return tokenizer

    def _generate_reference_text_batch(
        self,
        *,
        prompts: Sequence[str],
        max_new_tokens: int,
    ) -> List[str]:
        if not prompts:
            return []
        self._load()
        if self._reference_model is None:
            raise RuntimeError("reference answer fallback judge requires a loaded reference model")
        tokenizer = self._reference_answer_judge_tokenizer()
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": REFERENCE_ANSWER_JUDGE_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(prompt or "")}],
                },
            ]
            for prompt in prompts
        ]
        rendered = [
            tokenizer.apply_chat_template(message_batch, tokenize=False, add_generation_prompt=True)
            for message_batch in messages
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True)
        inputs = self._move_batch_to_device(inputs, model=self._reference_model)
        torch = self._torch
        assert torch is not None
        with torch.inference_mode():
            output_ids = self._reference_model.generate(
                **inputs,
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
                return_audio=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        generated = output_ids[:, inputs["input_ids"].shape[-1] :]
        return tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def judge_answer_equivalence_batch(self, requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requests:
            return []
        max_new_tokens = max(1, int(self.config.answer_fallback_max_new_tokens or 96))
        prompts = [
            build_reference_answer_judge_prompt(
                question=str(request.get("question") or ""),
                gt_answer=str(request.get("gt_answer") or ""),
                model_output=str(request.get("model_output") or ""),
                choices=list(request.get("choices") or []),
            )
            for request in requests
        ]
        raw_outputs = self._generate_reference_text_batch(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )
        decisions: List[Optional[Dict[str, Any]]] = [None for _ in requests]
        retry_indices: List[int] = []
        retry_prompts: List[str] = []

        for idx, (request, raw_output) in enumerate(zip(requests, raw_outputs)):
            decision = resolve_reference_answer_judge_output(
                raw=str(raw_output or ""),
                gt_answer=str(request.get("gt_answer") or ""),
                model_output=str(request.get("model_output") or ""),
                choices=list(request.get("choices") or []),
            )
            decisions[idx] = dict(decision, judge_raw_output=str(raw_output or ""), judge_retry_raw_output="")
            if not has_reference_answer_judge_json(str(raw_output or "")):
                retry_indices.append(idx)
                retry_prompts.append(
                    build_reference_answer_judge_retry_prompt(
                        question=str(request.get("question") or ""),
                        gt_answer=str(request.get("gt_answer") or ""),
                        model_output=str(request.get("model_output") or ""),
                        prior_raw=str(raw_output or ""),
                        choices=list(request.get("choices") or []),
                    )
                )

        if retry_prompts:
            retry_outputs = self._generate_reference_text_batch(
                prompts=retry_prompts,
                max_new_tokens=max_new_tokens,
            )
            for idx, retry_raw in zip(retry_indices, retry_outputs):
                request = requests[idx]
                resolved = resolve_reference_answer_judge_output(
                    raw=str(retry_raw or ""),
                    gt_answer=str(request.get("gt_answer") or ""),
                    model_output=str(request.get("model_output") or ""),
                    choices=list(request.get("choices") or []),
                )
                decisions[idx] = dict(
                    resolved,
                    judge_raw_output=str(decisions[idx].get("judge_raw_output") if decisions[idx] else ""),
                    judge_retry_raw_output=str(retry_raw or ""),
                )

        return [dict(item or {}) for item in decisions]

    def _optimizer_zero_grad(self) -> None:
        if self._deepspeed_engine is not None:
            self._deepspeed_engine.zero_grad()
            return
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)

    def _backward_loss(self, loss) -> None:
        if self._deepspeed_engine is not None:
            self._deepspeed_engine.backward(loss)
            return
        loss.backward()

    def _optimizer_step(self) -> None:
        if self._deepspeed_engine is not None:
            self._deepspeed_engine.step()
            return
        if self._optimizer is not None:
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

    def _compute_token_logprobs(self, model_inputs: Dict[str, Any], model) -> tuple[Any, Any, Any]:
        torch = self._torch
        labels = model_inputs["labels"]
        forward_inputs = {key: value for key, value in model_inputs.items() if key != "labels"}
        forward_model = self._forward_target_model(model)
        outputs = forward_model(**forward_inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_logits.device)
        token_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~token_mask, 0)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_entropies = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        token_logprobs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * token_mask.to(token_logprobs.dtype)
        token_entropies = token_entropies * token_mask.to(token_entropies.dtype)
        return token_logprobs, token_mask, token_entropies

    def _credit_assignment_mode(self) -> str:
        mode = str(self.config.credit_assignment or "").strip().lower()
        if mode in {"", "hybrid-local", "hybrid-controller", "local", "controller"}:
            return "hybrid-local"
        if mode in {"sequence", "rollout-scalar", "scalar"}:
            return "sequence"
        return mode

    def _normalize_scores(self, values: Sequence[float]) -> List[float]:
        if not values:
            return []
        mean = sum(float(value) for value in values) / float(len(values))
        variance = sum((float(value) - mean) ** 2 for value in values) / float(len(values))
        if variance <= 1e-12:
            return [0.0 for _ in values]
        std = variance ** 0.5
        return [(float(value) - mean) / std for value in values]

    def _local_process_signal_for_turn(
        self,
        turn_example,
        reward: Dict[str, Any],
    ) -> Tuple[str, float]:
        try:
            chunk_index = int(turn_example.chunk_index)
        except Exception:
            chunk_index = -1

        if turn_example.turn_type == "think":
            if bool(getattr(turn_example, "is_final_think", False)):
                if bool(reward.get("R_t_final_judged", 0.0)):
                    return "think", float(reward.get("R_t_final", 0.0))
            per_chunk = list(reward.get("R_t_per_chunk", []) or [])
            judged_mask = list(reward.get("R_t_judged_mask", []) or [])
            if 0 <= chunk_index < len(per_chunk):
                local_score = float(per_chunk[chunk_index])
                if 0 <= chunk_index < len(judged_mask) and bool(judged_mask[chunk_index]):
                    return "think", local_score
                if abs(local_score) > 1e-8:
                    return "think", local_score
            per_tick = list(reward.get("R_u_per_tick", []) or [])
            if 0 <= chunk_index < len(per_tick):
                return "think", float(per_tick[chunk_index])
            if abs(float(reward.get("R_t", 0.0))) > 1e-8:
                return "think", float(reward.get("R_t", 0.0))
            return "think", float(reward.get("R_u", 0.0))

        if turn_example.turn_type == "wait":
            per_tick = list(reward.get("R_u_per_tick", []) or [])
            if 0 <= chunk_index < len(per_tick):
                return "wait", float(per_tick[chunk_index])
            return "wait", float(reward.get("R_u", 0.0))

        return "", 0.0

    def _per_action_process_advantages(
        self,
        turn_examples_by_rollout: Sequence[Sequence[Any]],
        rewards: Sequence[Dict[str, Any]],
    ) -> List[List[float]]:
        if not turn_examples_by_rollout:
            return []

        normalized: List[List[float]] = [
            [0.0 for _ in turn_examples]
            for turn_examples in turn_examples_by_rollout
        ]

        action_scores: Dict[str, List[float]] = {}
        action_refs: List[Tuple[int, int, str]] = []

        for rollout_index, turn_examples in enumerate(turn_examples_by_rollout):
            reward = rewards[rollout_index] if rollout_index < len(rewards) else {}
            for turn_index, turn_example in enumerate(turn_examples):
                action_type, local_score = self._local_process_signal_for_turn(turn_example, reward)
                if not action_type:
                    continue
                bucket = "controller" if action_type in {"think", "wait"} else action_type
                action_scores.setdefault(bucket, []).append(float(local_score))
                action_refs.append((rollout_index, turn_index, bucket))

        if not action_refs:
            return normalized

        normalized_by_action = {
            action_type: self._normalize_scores(scores)
            for action_type, scores in action_scores.items()
        }
        action_offsets = {action_type: 0 for action_type in normalized_by_action}

        for rollout_index, turn_index, action_type in action_refs:
            offset = action_offsets[action_type]
            normalized[rollout_index][turn_index] = float(normalized_by_action[action_type][offset])
            action_offsets[action_type] = offset + 1

        return normalized

    def _iter_rollout_sequences(self, group_batches: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sequences: List[Dict[str, Any]] = []
        total_turn_examples = 0
        global_rollout_index = 0
        soft_cap = max(0, int(self.config.max_turn_examples or 0))
        credit_mode = self._credit_assignment_mode()
        hybrid_alpha = float(self.config.hybrid_alpha)

        for prompt_index, group_batch in enumerate(group_batches):
            sample = group_batch["sample"]
            rollouts = list(group_batch.get("rollouts", []) or [])
            rewards = list(group_batch.get("rewards", []) or [])
            advantages = list(group_batch.get("advantages", []) or [])
            group_entries: List[Dict[str, Any]] = []
            for local_rollout_index, rollout in enumerate(rollouts):
                rollout_global_index = global_rollout_index
                turn_examples = build_turn_training_examples(
                    sample,
                    rollout,
                    prompt_index=prompt_index,
                    rollout_index=rollout_global_index,
                )
                global_rollout_index += 1
                if not turn_examples:
                    continue
                group_entries.append(
                    {
                        "local_rollout_index": local_rollout_index,
                        "global_rollout_index": rollout_global_index,
                        "turn_examples": turn_examples,
                    }
                )

            process_advantages = (
                self._per_action_process_advantages(
                    [entry["turn_examples"] for entry in group_entries],
                    [
                        rewards[entry["local_rollout_index"]]
                        if entry["local_rollout_index"] < len(rewards)
                        else {}
                        for entry in group_entries
                    ],
                )
                if credit_mode == "hybrid-local"
                else []
            )

            for group_entry_index, entry in enumerate(group_entries):
                turn_examples = entry["turn_examples"]
                if soft_cap > 0 and sequences and total_turn_examples + len(turn_examples) > soft_cap:
                    return sequences
                local_rollout_index = entry["local_rollout_index"]
                outcome_advantage = float(advantages[local_rollout_index]) if local_rollout_index < len(advantages) else 0.0
                turn_advantages: List[float] = []
                for turn_position, turn_example in enumerate(turn_examples):
                    if credit_mode == "hybrid-local" and turn_example.turn_type in {"think", "wait"}:
                        process_advantage = 0.0
                        if group_entry_index < len(process_advantages) and turn_position < len(process_advantages[group_entry_index]):
                            process_advantage = float(process_advantages[group_entry_index][turn_position])
                        turn_advantages.append(
                            hybrid_alpha * outcome_advantage + (1.0 - hybrid_alpha) * process_advantage
                        )
                    else:
                        turn_advantages.append(outcome_advantage)
                sequences.append(
                    {
                        "sample": sample,
                        "prompt_index": prompt_index,
                        "rollout_index": entry["global_rollout_index"],
                        "local_rollout_index": local_rollout_index,
                        "turn_examples": turn_examples,
                        "turn_advantages": turn_advantages,
                        "outcome_advantage": outcome_advantage,
                        "reward_total": float(rewards[local_rollout_index].get("total", 0.0)) if local_rollout_index < len(rewards) else 0.0,
                        "reward_outcome": float(
                            rewards[local_rollout_index].get(
                                "R_outcome",
                                rewards[local_rollout_index].get("total", 0.0),
                            )
                        ) if local_rollout_index < len(rewards) else 0.0,
                    }
                )
                total_turn_examples += len(turn_examples)
        return sequences

    def _turn_batches(self, turn_examples: Sequence[Any]) -> Iterable[List[Any]]:
        batch_size = max(1, int(self.config.turn_batch_size or 1))
        for start in range(0, len(turn_examples), batch_size):
            yield list(turn_examples[start : start + batch_size])

    def _annotate_behavior_logprobs(
        self,
        turn_records: Sequence[Dict[str, Any]],
        *,
        audio_cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not turn_records:
            return
        torch = self._torch
        for turn_batch_records in self._turn_batches(turn_records):
            turn_batch = [record["example"] for record in turn_batch_records]
            example_inputs = prepare_turn_model_inputs_batch(
                self._processor,
                turn_batch,
                sampling_rate=self.config.sampling_rate,
                audio_cache=audio_cache,
            )
            policy_inputs = self._move_batch_to_device(example_inputs, model=self._model)
            with torch.no_grad():
                token_logprobs, token_mask, _token_entropies = self._compute_token_logprobs(policy_inputs, self._model)
                sequence_logprobs = normalized_sequence_logprob_from_token_logprobs(
                    token_logprobs,
                    token_mask=token_mask,
                )
            token_logprobs_cpu = token_logprobs.detach().cpu()
            for record, seq_logprob, token_row in zip(turn_batch_records, sequence_logprobs.tolist(), token_logprobs_cpu):
                record["behavior_logprob"] = float(seq_logprob)
                if str(self.config.loss_mode or "grpo-sequence").strip().lower() == "dapo-token":
                    record["behavior_token_logprobs"] = [float(value) for value in token_row.tolist()]

    def _build_behavior_token_logprobs(
        self,
        turn_batch_records: Sequence[Dict[str, Any]],
        *,
        like: Any,
    ):
        torch = self._torch
        behavior = like.new_zeros(like.shape)
        width = int(like.shape[-1]) if like.ndim >= 2 else 0
        for row_index, record in enumerate(turn_batch_records):
            values = list(record.get("behavior_token_logprobs", []) or [])
            if not values or width <= 0:
                continue
            trimmed = values[:width]
            behavior[row_index, : len(trimmed)] = torch.tensor(trimmed, dtype=like.dtype, device=like.device)
        return behavior

    def update_groups(self, group_batches: Sequence[Dict[str, Any]], step_index: Optional[int] = None) -> Dict[str, Any]:
        self._load()
        torch = self._torch

        if not group_batches:
            return {
                "mode": "trainable",
                "backend": "omni-vllm",
                "credit_assignment": self._credit_assignment_mode(),
                "optimizer": self._optimizer_kind,
                "n_prompt_groups": 0,
                "n_rollouts": 0,
                "n_turn_examples": 0,
            }

        rollout_sequences = self._iter_rollout_sequences(group_batches)
        if not rollout_sequences:
            return {
                "mode": "trainable",
                "backend": "omni-vllm",
                "credit_assignment": self._credit_assignment_mode(),
                "optimizer": self._optimizer_kind,
                "n_prompt_groups": len(group_batches),
                "n_rollouts": 0,
                "n_turn_examples": 0,
            }

        turn_records = [
            {"example": example, "advantage": float(turn_advantage), "sample": rollout_sequence.get("sample")}
            for rollout_sequence in rollout_sequences
            for example, turn_advantage in zip(
                rollout_sequence["turn_examples"],
                rollout_sequence["turn_advantages"],
            )
        ]

        log_prefix = "[omni-updater]"
        LOGGER.info(
            "%s step=%s prompt_groups=%s rollouts=%s turn_records=%s turn_batch_size=%s ds=%s loss_mode=%s",
            log_prefix,
            int(step_index or 0),
            len(group_batches),
            len(rollout_sequences),
            len(turn_records),
            int(self.config.turn_batch_size or 1),
            bool(self._deepspeed_engine is not None),
            str(self.config.loss_mode or "grpo-sequence"),
        )

        total_tokens = 0
        turn_count = 0
        forward_batch_count = 0
        audio_cache = {} if self.config.cache_audio_arrays else None
        n_rollouts = len(rollout_sequences)
        mean_advantage = sum(float(rollout_sequence["outcome_advantage"]) for rollout_sequence in rollout_sequences)
        mean_reward = sum(float(rollout_sequence["reward_total"]) for rollout_sequence in rollout_sequences)
        mean_outcome = sum(float(rollout_sequence["reward_outcome"]) for rollout_sequence in rollout_sequences)
        total_objective_value = 0.0
        total_entropy_value = 0.0
        total_entropy_tokens = 0
        total_clip_fraction_value = 0.0
        total_clip_fraction_batches = 0
        total_observed_kl_value = 0.0
        total_observed_kl_batches = 0
        total_reference_kl_value = 0.0
        total_reference_kl_batches = 0
        total_overlong_penalty_value = 0.0
        total_overlong_penalty_batches = 0
        total_overlong_fraction_value = 0.0
        total_overlong_fraction_batches = 0
        total_completion_tokens = 0.0
        total_completion_token_batches = 0
        loss_mode = str(self.config.loss_mode or "grpo-sequence").strip().lower()
        lr_scale = self._set_learning_rate_for_step(step_index)
        update_epochs = max(1, int(self.config.update_epochs or 1))
        precompute_behavior_logprobs = update_epochs > 1
        if precompute_behavior_logprobs:
            self._annotate_behavior_logprobs(turn_records, audio_cache=audio_cache)

        grad_norms: List[float] = []
        epoch_objective_values: List[float] = []
        for _epoch_index in range(update_epochs):
            self._optimizer_zero_grad()
            epoch_objective_value = 0.0
            epoch_total_tokens = 0
            epoch_turn_count = 0
            epoch_forward_batches = 0
            total_epoch_batches = max(
                1,
                (len(turn_records) + max(1, int(self.config.turn_batch_size or 1)) - 1)
                // max(1, int(self.config.turn_batch_size or 1)),
            )

            for turn_batch_records in self._turn_batches(turn_records):
                turn_batch = [record["example"] for record in turn_batch_records]
                example_inputs = prepare_turn_model_inputs_batch(
                    self._processor,
                    turn_batch,
                    sampling_rate=self.config.sampling_rate,
                    audio_cache=audio_cache,
                )
                policy_inputs = self._move_batch_to_device(example_inputs, model=self._model)
                behavior_sequence_logprobs = None
                behavior_token_logprobs = None
                if not precompute_behavior_logprobs:
                    with torch.no_grad():
                        behavior_token_logprobs, behavior_token_mask, _behavior_token_entropies = self._compute_token_logprobs(
                            policy_inputs,
                            self._model,
                        )
                        behavior_sequence_logprobs = normalized_sequence_logprob_from_token_logprobs(
                            behavior_token_logprobs,
                            token_mask=behavior_token_mask,
                        ).detach()
                    if loss_mode == "dapo-token":
                        behavior_token_logprobs = behavior_token_logprobs.detach()
                    else:
                        behavior_token_logprobs = None
                policy_token_logprobs, token_mask, token_entropies = self._compute_token_logprobs(policy_inputs, self._model)
                batch_device = policy_token_logprobs.device
                batch_sequence_logprobs = normalized_sequence_logprob_from_token_logprobs(
                    policy_token_logprobs,
                    token_mask=token_mask,
                )

                batch_sequence_kls = None
                if self._reference_model is not None and self.config.kl_beta > 0:
                    with torch.no_grad():
                        reference_inputs = self._move_batch_to_device(example_inputs, model=self._reference_model)
                        reference_token_logprobs, _reference_mask, _reference_entropies = self._compute_token_logprobs(
                            reference_inputs,
                            self._reference_model,
                        )
                    reference_token_logprobs = reference_token_logprobs.to(batch_device)
                    batch_sequence_kls = approximate_reverse_kl(
                        policy_token_logprobs,
                        reference_token_logprobs,
                        token_mask=token_mask,
                    )

                batch_advantages = batch_sequence_logprobs.new_tensor(
                    [float(record["advantage"]) for record in turn_batch_records]
                )
                if behavior_sequence_logprobs is None:
                    behavior_sequence_logprobs = batch_sequence_logprobs.new_tensor(
                        [float(record.get("behavior_logprob", 0.0)) for record in turn_batch_records]
                    )
                else:
                    behavior_sequence_logprobs = behavior_sequence_logprobs.to(
                        device=batch_device,
                        dtype=batch_sequence_logprobs.dtype,
                    )
                if loss_mode == "dapo-token":
                    if behavior_token_logprobs is None:
                        behavior_token_logprobs = self._build_behavior_token_logprobs(
                            turn_batch_records,
                            like=policy_token_logprobs,
                        )
                    else:
                        behavior_token_logprobs = behavior_token_logprobs.to(
                            device=batch_device,
                            dtype=policy_token_logprobs.dtype,
                        )
                    batch_clip_fraction = clip_fraction_from_token_logprobs(
                        policy_token_logprobs,
                        behavior_token_logprobs,
                        token_mask=token_mask,
                        epsilon_low=float(self.config.epsilon_low),
                        epsilon_high=float(self.config.epsilon_high),
                    )
                    batch_observed_kl = approximate_observed_kl_from_token_logprobs(
                        policy_token_logprobs,
                        behavior_token_logprobs,
                        token_mask=token_mask,
                    )
                    batch_objective = clipped_dapo_token_objective(
                        policy_token_logprobs=policy_token_logprobs,
                        behavior_token_logprobs=behavior_token_logprobs,
                        advantages=batch_advantages,
                        token_mask=token_mask,
                        epsilon_low=float(self.config.epsilon_low),
                        epsilon_high=float(self.config.epsilon_high),
                    )
                    completion_lengths = token_mask.to(dtype=policy_token_logprobs.dtype).sum(dim=-1)
                    batch_mean_completion_tokens = completion_lengths.mean()
                    if self.config.overlong_shaping:
                        batch_overlong_penalty = overlong_shaping_penalty(
                            token_mask=token_mask,
                            threshold_tokens=int(self.config.overlong_threshold_tokens),
                            penalty_slope=float(self.config.overlong_penalty_slope),
                            penalty_cap=float(self.config.overlong_penalty_cap),
                        ).to(dtype=batch_objective.dtype, device=batch_objective.device)
                    else:
                        batch_overlong_penalty = torch.zeros_like(batch_objective)
                    batch_overlong_fraction = completion_lengths.gt(
                        float(self.config.overlong_threshold_tokens)
                    ).to(dtype=policy_token_logprobs.dtype).mean()
                    batch_objective = batch_objective - batch_overlong_penalty
                else:
                    batch_clip_fraction = clip_fraction_from_sequence_logprobs(
                        batch_sequence_logprobs,
                        behavior_sequence_logprobs,
                        epsilon_low=float(self.config.epsilon_low),
                        epsilon_high=float(self.config.epsilon_high),
                    )
                    batch_observed_kl = approximate_observed_kl_from_sequence_logprobs(
                        batch_sequence_logprobs,
                        behavior_sequence_logprobs,
                    )
                    batch_objective = clipped_grpo_objective(
                        policy_token_logprobs=policy_token_logprobs,
                        behavior_sequence_logprobs=behavior_sequence_logprobs,
                        advantages=batch_advantages,
                        token_mask=token_mask,
                        epsilon_low=float(self.config.epsilon_low),
                        epsilon_high=float(self.config.epsilon_high),
                    )
                    batch_mean_completion_tokens = token_mask.to(dtype=policy_token_logprobs.dtype).sum(dim=-1).mean()
                    batch_overlong_penalty = torch.zeros_like(batch_objective)
                    batch_overlong_fraction = torch.tensor(
                        0.0,
                        dtype=policy_token_logprobs.dtype,
                        device=policy_token_logprobs.device,
                    )
                if batch_sequence_kls is not None:
                    batch_objective = batch_objective - float(self.config.kl_beta) * batch_sequence_kls
                batch_loss = -batch_objective.sum() / float(max(1, n_rollouts * update_epochs))
                self._backward_loss(batch_loss)
                epoch_objective_value += float(batch_objective.detach().sum().cpu().item())
                total_clip_fraction_value += float(batch_clip_fraction.detach().cpu().item())
                total_clip_fraction_batches += 1
                total_observed_kl_value += float(batch_observed_kl.detach().cpu().item())
                total_observed_kl_batches += 1
                total_entropy_value += float(token_entropies.detach().sum().cpu().item())
                total_entropy_tokens += int(token_mask.sum().item())
                total_overlong_penalty_value += float(batch_overlong_penalty.detach().mean().cpu().item())
                total_overlong_penalty_batches += 1
                total_overlong_fraction_value += float(batch_overlong_fraction.detach().cpu().item())
                total_overlong_fraction_batches += 1
                total_completion_tokens += float(batch_mean_completion_tokens.detach().cpu().item())
                total_completion_token_batches += 1
                if batch_sequence_kls is not None:
                    total_reference_kl_value += float(batch_sequence_kls.detach().mean().cpu().item())
                    total_reference_kl_batches += 1

                epoch_total_tokens += int(token_mask.sum().item())
                epoch_turn_count += len(turn_batch_records)
                epoch_forward_batches += 1
                if (
                    epoch_forward_batches == 1
                    or epoch_forward_batches == total_epoch_batches
                    or epoch_forward_batches % 25 == 0
                ):
                    LOGGER.info(
                        "%s step=%s epoch=%s/%s turn_batch=%s/%s batch_examples=%s cumulative_tokens=%s observed_kl=%.4f clip=%.4f",
                        log_prefix,
                        int(step_index or 0),
                        int(_epoch_index + 1),
                        int(update_epochs),
                        int(epoch_forward_batches),
                        int(total_epoch_batches),
                        len(turn_batch_records),
                        int(epoch_total_tokens),
                        float(batch_observed_kl.detach().cpu().item()),
                        float(batch_clip_fraction.detach().cpu().item()),
                    )

            if epoch_turn_count == 0:
                continue

            grad_norm = None
            if self._deepspeed_engine is None and self.config.grad_clip_norm and self.config.grad_clip_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), self.config.grad_clip_norm).item()
                )
                grad_norms.append(grad_norm)

            self._optimizer_step()

            epoch_objective_values.append(epoch_objective_value)
            total_objective_value += epoch_objective_value
            total_tokens += epoch_total_tokens
            turn_count = max(turn_count, epoch_turn_count)
            forward_batch_count += epoch_forward_batches

        if turn_count == 0:
            return {
                "mode": "trainable",
                "backend": "omni-vllm",
                "credit_assignment": self._credit_assignment_mode(),
                "optimizer": self._optimizer_kind,
                "n_prompt_groups": len(group_batches),
                "n_rollouts": 0,
                "n_turn_examples": turn_count,
            }

        mean_loss_value = -total_objective_value / float(max(1, n_rollouts * update_epochs))
        grad_norm = (sum(grad_norms) / float(len(grad_norms))) if grad_norms else None
        observed_entropy = (total_entropy_value / float(total_entropy_tokens)) if total_entropy_tokens else None
        clip_fraction = (total_clip_fraction_value / float(total_clip_fraction_batches)) if total_clip_fraction_batches else None
        observed_kl = (total_observed_kl_value / float(total_observed_kl_batches)) if total_observed_kl_batches else None
        reference_kl = (total_reference_kl_value / float(total_reference_kl_batches)) if total_reference_kl_batches else None
        overlong_penalty = (
            total_overlong_penalty_value / float(total_overlong_penalty_batches)
            if total_overlong_penalty_batches
            else None
        )
        overlong_fraction = (
            total_overlong_fraction_value / float(total_overlong_fraction_batches)
            if total_overlong_fraction_batches
            else None
        )
        mean_completion_tokens = (
            total_completion_tokens / float(total_completion_token_batches)
            if total_completion_token_batches
            else None
        )

        n_rollouts = len(rollout_sequences)
        return {
            "mode": "trainable",
            "backend": "omni-vllm",
            "actor_model": self.config.model_name,
            "credit_assignment": self._credit_assignment_mode(),
            "optimizer": self._optimizer_kind,
            "optimizer_summary": dict(self._optimizer_summary),
            "deepspeed_enabled": bool(self._deepspeed_engine is not None),
            "deepspeed_zero_stage": int(self.config.deepspeed_zero_stage or 0),
            "n_prompt_groups": len(group_batches),
            "n_rollouts": n_rollouts,
            "n_turn_examples": turn_count,
            "n_forward_batches": forward_batch_count,
            "supervised_tokens": total_tokens,
            "mean_loss": float(mean_loss_value),
            "mean_advantage": mean_advantage / float(max(1, n_rollouts)),
            "mean_reward": mean_reward / float(max(1, n_rollouts)),
            "mean_outcome": mean_outcome / float(max(1, n_rollouts)),
            "grad_norm": grad_norm,
            "entropy": observed_entropy,
            "observed_kl": observed_kl,
            "reference_kl": reference_kl,
            "clip_fraction": clip_fraction,
            "overlong_penalty": overlong_penalty,
            "overlong_fraction": overlong_fraction,
            "mean_completion_tokens": mean_completion_tokens,
            "kl_beta": float(self.config.kl_beta),
            "hybrid_alpha": float(self.config.hybrid_alpha),
            "update_epochs": int(update_epochs),
            "epsilon_low": float(self.config.epsilon_low),
            "epsilon_high": float(self.config.epsilon_high),
            "overlong_shaping": bool(self.config.overlong_shaping),
            "overlong_threshold_tokens": int(self.config.overlong_threshold_tokens),
            "loss_mode": loss_mode,
            "lr_scale": float(lr_scale),
            "resume_checkpoint": str(self.config.resume_checkpoint or ""),
        }

    def update_group(
        self,
        sample,
        rollouts: List[Any],
        rewards: List[Dict[str, float]],
        advantages: List[float],
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.update_groups(
            [
                {
                    "sample": sample,
                    "rollouts": list(rollouts),
                    "rewards": list(rewards),
                    "advantages": list(advantages),
                }
            ],
            step_index=step_index,
        )

    def save_checkpoint(self, path: str, step: int, checkpoint_mode: Optional[str] = None) -> CheckpointArtifact:
        self._load()

        checkpoint_path = Path(path)
        checkpoint_dir = checkpoint_path.with_suffix("") if checkpoint_path.suffix else checkpoint_path
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_mode = str(checkpoint_mode or self.config.checkpoint_mode or "full").lower()
        if checkpoint_mode not in {"full", "model-only", "metadata-only"}:
            raise ValueError("Unsupported checkpoint mode: {}".format(self.config.checkpoint_mode))

        extra_files: Dict[str, str] = {}
        lora_metadata = self._lora_metadata_payload()
        reloadable_model_path = ""
        if checkpoint_mode in {"full", "model-only"}:
            adapter_dir: Optional[Path] = None
            if self._using_lora:
                adapter_dir = self._save_lora_adapter(checkpoint_dir / self.config.lora_adapter_dirname)
                lora_metadata = self._lora_metadata_payload(adapter_subdir=adapter_dir.name)
                if checkpoint_mode == "full":
                    self._export_merged_lora_model(checkpoint_dir, adapter_dir)
                reloadable_model_path = str(adapter_dir)
            else:
                self._model.save_pretrained(checkpoint_dir)
                reloadable_model_path = str(checkpoint_dir)
            self._processor.save_pretrained(checkpoint_dir)
            if checkpoint_mode == "full" or not self._using_lora:
                extra_files = self._copy_required_checkpoint_files(checkpoint_dir)

        optimizer_path = ""
        deepspeed_metadata: Dict[str, Any] = {}
        if checkpoint_mode == "full" and self._optimizer is not None:
            if self._deepspeed_engine is not None:
                ds_root = self._deepspeed_checkpoint_root(checkpoint_dir)
                ds_root.mkdir(parents=True, exist_ok=True)
                tag = self._deepspeed_tag_for_step(step)
                self._deepspeed_engine.save_checkpoint(str(ds_root), tag=tag)
                optimizer_path = str(ds_root / tag)
                deepspeed_metadata = {
                    "enabled": True,
                    "checkpoint_subdir": ds_root.name,
                    "tag": tag,
                    "zero_stage": int(self.config.deepspeed_zero_stage or 0),
                }
            else:
                optimizer_path = str(checkpoint_dir / "optimizer.pt")
                self._torch.save(self._optimizer.state_dict(), optimizer_path)

        metadata = {
            "step": step,
            "config": asdict(self.config),
            "checkpoint_mode": checkpoint_mode,
            "extra_files": extra_files,
            "lora": lora_metadata,
            "deepspeed": deepspeed_metadata,
            "optimizer": {
                "name": self._optimizer_kind,
                "summary": dict(self._optimizer_summary),
            },
        }
        metadata_path = checkpoint_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return CheckpointArtifact(
            step=step,
            mode=checkpoint_mode,
            checkpoint_path=str(checkpoint_path),
            checkpoint_dir=str(checkpoint_dir),
            metadata_path=str(metadata_path),
            optimizer_path=optimizer_path,
            reloadable_model_path=reloadable_model_path,
            extra=extra_files,
        )
