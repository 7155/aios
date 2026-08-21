"""Local A/B inference service for the AIOS-IME browser frontend.

Each comparison slot owns a separate spawned process.  This is intentional:
AIOS keeps one CUDA execution context in process-global state, so loading two
models in the HTTP process would make the second ``LLM`` initialization fail.
The parent executes A and B sequentially to keep latency comparisons honest on
one local GPU.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .ime import ImeGenerationConfig


ATTNRES_BACKENDS = {"default", "reference", "eager", "compiled", "triton"}
SLOT_IDS = ("a", "b")
DEFAULT_EXAMPLES = (
    "机器学习是",
    "深度学习的核心是",
    "没关系，你先忙你的，",
    "今天的任务先做到这里，剩下的",
    "监督数据还不完整，需要",
    "先把重复数据删掉，再",
    "刚才我翻到我们以前一起拍的照片，有几张现在看还是挺有意思",
)


def _bounded_int(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是整数") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum}～{maximum} 之间")
    return result


def _bounded_float(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是数字") from error
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} 必须在 {minimum:g}～{maximum:g} 之间")
    return result


def json_safe(value: Any) -> Any:
    """Replace non-finite floats before serializing browser API responses."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_path: str
    backend: str = "default"
    kv_cache_max_tokens: int = 512
    attention_workspace_mib: int = 8
    device: str = "cuda:0"

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        require_local_path: bool = True,
    ) -> "ModelSpec":
        if not isinstance(payload, dict):
            raise ValueError("模型配置必须是对象")
        label = str(payload.get("label", "")).strip()
        if not label:
            raise ValueError("模型名称不能为空")
        if len(label) > 80:
            raise ValueError("模型名称不能超过 80 个字符")

        raw_path = str(payload.get("model_path", "")).strip()
        if not raw_path:
            raise ValueError(f"{label} 的模型目录不能为空")
        if require_local_path:
            model_path = str(Path(raw_path).expanduser().resolve())
            if not Path(model_path).is_dir():
                raise ValueError(f"模型目录不存在：{model_path}")
            if not (Path(model_path) / "config.json").is_file():
                raise ValueError(f"模型目录缺少 config.json：{model_path}")
        else:
            model_path = raw_path

        backend = str(payload.get("backend", "default")).strip().lower()
        if backend not in ATTNRES_BACKENDS:
            raise ValueError(f"不支持的 AttnRes backend：{backend}")
        device = str(payload.get("device", "cuda:0")).strip()
        if not device.startswith("cuda"):
            raise ValueError("AIOS-IME 前端目前只支持 CUDA device")
        return cls(
            label=label,
            model_path=model_path,
            backend=backend,
            kv_cache_max_tokens=_bounded_int(
                payload.get("kv_cache_max_tokens", 512),
                "KV Cache token 数",
                64,
                8192,
            ),
            attention_workspace_mib=_bounded_int(
                payload.get("attention_workspace_mib", 8),
                "Attention workspace",
                1,
                256,
            ),
            device=device,
        )

    @property
    def runtime_key(self) -> tuple[Any, ...]:
        return (
            self.model_path,
            self.backend,
            self.kv_cache_max_tokens,
            self.attention_workspace_mib,
            self.device,
        )


@dataclass(frozen=True)
class GenerationSettings:
    display_candidates: int = 3
    sampling_attempts: int = 8
    max_sampling_attempts: int = 24
    refill_batch_size: int = 8
    min_refill_batch_size: int = 2
    max_new_tokens: int = 12
    min_new_tokens: int = 2
    max_candidate_chars: int = 32
    temperature: float = 0.35
    top_k: int = 50
    top_p: float = 0.9
    refill_temperature: float = 0.75
    refill_top_k: int = 96
    refill_top_p: float = 0.95
    diversity_lambda: float = 0.35
    seed: int = 20260814

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "GenerationSettings":
        payload = payload or {}
        if not isinstance(payload, dict):
            raise ValueError("生成参数必须是对象")
        settings = cls(
            display_candidates=3,
            sampling_attempts=_bounded_int(
                payload.get("sampling_attempts", 8), "首轮候选路数", 3, 32
            ),
            max_sampling_attempts=_bounded_int(
                payload.get("max_sampling_attempts", 24), "候选总预算", 3, 64
            ),
            refill_batch_size=_bounded_int(
                payload.get("refill_batch_size", 8), "每轮补采样路数", 1, 16
            ),
            min_refill_batch_size=_bounded_int(
                payload.get("min_refill_batch_size", 2), "最小补采样路数", 1, 16
            ),
            max_new_tokens=_bounded_int(
                payload.get("max_new_tokens", 12), "最大输出 token", 1, 64
            ),
            min_new_tokens=_bounded_int(
                payload.get("min_new_tokens", 2), "最小输出 token", 0, 64
            ),
            max_candidate_chars=_bounded_int(
                payload.get("max_candidate_chars", 32), "候选最大字符数", 4, 128
            ),
            temperature=_bounded_float(
                payload.get("temperature", 0.35), "Temperature", 0.01, 2.0
            ),
            top_k=_bounded_int(payload.get("top_k", 50), "Top-k", 1, 512),
            top_p=_bounded_float(payload.get("top_p", 0.9), "Top-p", 0.01, 1.0),
            refill_temperature=_bounded_float(
                payload.get("refill_temperature", 0.75),
                "补采样 Temperature",
                0.01,
                2.0,
            ),
            refill_top_k=_bounded_int(
                payload.get("refill_top_k", 96), "补采样 Top-k", 1, 512
            ),
            refill_top_p=_bounded_float(
                payload.get("refill_top_p", 0.95), "补采样 Top-p", 0.01, 1.0
            ),
            diversity_lambda=_bounded_float(
                payload.get("diversity_lambda", 0.35), "多样性系数", 0.0, 2.0
            ),
            seed=_bounded_int(payload.get("seed", 20260814), "随机种子", 0, 2**63 - 1),
        )
        if settings.max_sampling_attempts < settings.sampling_attempts:
            raise ValueError("候选总预算不能小于首轮候选路数")
        if settings.min_refill_batch_size > settings.refill_batch_size:
            raise ValueError("最小补采样路数不能大于每轮补采样路数")
        if settings.min_new_tokens > settings.max_new_tokens:
            raise ValueError("最小输出 token 不能大于最大输出 token")
        settings.to_ime_config().validate()
        return settings

    def to_ime_config(self) -> ImeGenerationConfig:
        return ImeGenerationConfig(**asdict(self))


@dataclass(frozen=True)
class CompareRequest:
    prefix: str
    slots: dict[str, ModelSpec]
    generation: GenerationSettings
    targets: tuple[str, ...]
    order: Literal["a_then_b", "b_then_a"]
    reset_prefix_cache: bool

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        require_local_paths: bool,
    ) -> "CompareRequest":
        if not isinstance(payload, dict):
            raise ValueError("请求必须是 JSON 对象")
        prefix = str(payload.get("prefix", ""))
        if not prefix.strip():
            raise ValueError("请输入中文前缀")
        if len(prefix) > 512:
            raise ValueError("前缀不能超过 512 个字符")

        raw_slots = payload.get("slots")
        if not isinstance(raw_slots, dict):
            raise ValueError("请求缺少 A/B 模型配置")
        slots = {
            slot_id: ModelSpec.from_payload(
                raw_slots.get(slot_id, {}),
                require_local_path=require_local_paths,
            )
            for slot_id in SLOT_IDS
        }

        raw_targets = payload.get("targets", list(SLOT_IDS))
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError("至少选择一个运行目标")
        targets = tuple(dict.fromkeys(str(item).lower() for item in raw_targets))
        if any(item not in SLOT_IDS for item in targets):
            raise ValueError("运行目标只能是 a 或 b")

        order = str(payload.get("order", "a_then_b"))
        if order not in {"a_then_b", "b_then_a"}:
            raise ValueError("运行顺序只能是 a_then_b 或 b_then_a")
        reset_value = payload.get("reset_prefix_cache", False)
        if not isinstance(reset_value, bool):
            raise ValueError("reset_prefix_cache 必须是布尔值")
        return cls(
            prefix=prefix,
            slots=slots,
            generation=GenerationSettings.from_payload(payload.get("generation")),
            targets=targets,
            order=order,
            reset_prefix_cache=reset_value,
        )


def _model_metadata(llm: Any, parameter_count: int) -> dict[str, Any]:
    config = llm.config
    attention_backend = llm.model.attn_backend
    return {
        "model_type": config.model_type,
        "architecture_revision": config.architecture_revision,
        "residual_type": config.residual_type,
        "attnres_backend": config.attnres_backend,
        "attention_backend": getattr(
            attention_backend,
            "backend_name",
            type(attention_backend).__name__,
        ),
        "layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "context_length": config.max_position_embeddings,
        "parameter_count": parameter_count,
        "dtype": str(llm.dtype).removeprefix("torch."),
    }


def _worker_main(connection: Any, spec_payload: dict[str, Any]) -> None:
    """Load one model and serve inference commands inside a CUDA-owning process."""
    try:
        import torch

        from aios import ImeCompletionEngine, ImeGenerationConfig, LLM

        spec = ModelSpec(**spec_payload)
        llm_kwargs: dict[str, Any] = {
            "device": spec.device,
            "kv_cache_max_tokens": spec.kv_cache_max_tokens,
            "attention_workspace_size": spec.attention_workspace_mib * 2**20,
        }
        if spec.backend != "default":
            llm_kwargs["attnres_backend"] = spec.backend
        load_started = time.perf_counter()
        llm = LLM(spec.model_path, **llm_kwargs)
        engine = ImeCompletionEngine(llm)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        # Compile/load lazy CUDA kernels before the first measured user request.
        # The worker startup may take longer, but ImeCompletionResult.latency_ms
        # then represents warm steady-state inference instead of one-time JIT.
        warmup_started = time.perf_counter()
        engine.complete(
            "这是预热输入",
            ImeGenerationConfig(
                sampling_attempts=8,
                max_sampling_attempts=8,
                refill_batch_size=8,
                max_new_tokens=2,
                min_new_tokens=0,
                seed=0,
            ),
        )
        engine.reset_prefix_cache()
        torch.cuda.synchronize(llm.device)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        # AIOS models inherit the lightweight BaseOP rather than nn.Module.
        # Its state_dict contains each loaded tensor once (including tied
        # embedding only once), which is the correct serving parameter count.
        parameter_count = sum(tensor.numel() for tensor in llm.model.state_dict().values())
        metadata = _model_metadata(llm, parameter_count)
        connection.send(
            {
                "kind": "ready",
                "load_ms": load_ms,
                "warmup_ms": warmup_ms,
                "metadata": metadata,
            }
        )
        request_index = 0

        while True:
            command = connection.recv()
            command_type = command.get("type")
            if command_type == "close":
                engine.reset_prefix_cache()
                connection.send({"kind": "closed"})
                break
            if command_type == "reset":
                engine.reset_prefix_cache()
                connection.send({"kind": "reset"})
                continue
            if command_type != "infer":
                connection.send(
                    {"kind": "error", "message": f"未知 worker 命令：{command_type}"}
                )
                continue

            try:
                if command.get("reset_prefix_cache", False):
                    engine.reset_prefix_cache()
                config = ImeGenerationConfig(**command["generation"])
                torch.cuda.reset_peak_memory_stats(llm.device)
                api_started = time.perf_counter()
                result = engine.complete(command["prefix"], config)
                torch.cuda.synchronize(llm.device)
                api_wall_ms = (time.perf_counter() - api_started) * 1000.0
                active_tokens_per_second = (
                    result.active_model_tokens * 1000.0 / result.latency_ms
                    if result.latency_ms > 0.0
                    else 0.0
                )
                gpu_active_tokens_per_second = (
                    result.active_model_tokens * 1000.0 / result.gpu_latency_ms
                    if result.gpu_latency_ms > 0.0
                    else 0.0
                )
                response = result.to_dict()
                response["runtime"] = {
                    "label": spec.label,
                    "model_path": spec.model_path,
                    "configured_backend": spec.backend,
                    "effective_backend": metadata["attnres_backend"],
                    "model_load_ms": load_ms,
                    "model_warmup_ms": warmup_ms,
                    "api_wall_ms": api_wall_ms,
                    "active_tokens_per_second": active_tokens_per_second,
                    "gpu_active_tokens_per_second": gpu_active_tokens_per_second,
                    "cuda_allocated_mib": torch.cuda.memory_allocated(llm.device) / 2**20,
                    "cuda_reserved_mib": torch.cuda.memory_reserved(llm.device) / 2**20,
                    "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(llm.device)
                    / 2**20,
                    "cold_request": False,
                    "first_request": request_index == 0,
                    "request_index": request_index + 1,
                    "model": metadata,
                }
                request_index += 1
                connection.send({"kind": "result", "result": response})
            except Exception as error:  # keep the loaded worker alive after bad input
                connection.send(
                    {
                        "kind": "error",
                        "message": f"{type(error).__name__}: {error}",
                    }
                )
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as error:
        try:
            connection.send(
                {
                    "kind": "startup_error",
                    "message": f"{type(error).__name__}: {error}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class InferenceWorker:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        startup_timeout: float = 180.0,
        inference_timeout: float = 180.0,
    ) -> None:
        self.spec = spec
        self.startup_timeout = startup_timeout
        self.inference_timeout = inference_timeout
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any | None = None
        self._process: multiprocessing.Process | None = None
        self.ready_info: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main,
            args=(child, asdict(self.spec)),
            name=f"aios-ime-{self.spec.label}",
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        if not parent.poll(self.startup_timeout):
            self.shutdown(force=True)
            raise TimeoutError(f"{self.spec.label} 模型加载超过 {self.startup_timeout:g} 秒")
        message = parent.recv()
        if message.get("kind") != "ready":
            self.shutdown(force=True)
            raise RuntimeError(message.get("message", "模型 worker 启动失败"))
        self.ready_info = message

    def infer(
        self,
        prefix: str,
        generation: GenerationSettings,
        *,
        reset_prefix_cache: bool,
    ) -> dict[str, Any]:
        with self._lock:
            self.start()
            assert self._connection is not None
            assert self._process is not None
            self._connection.send(
                {
                    "type": "infer",
                    "prefix": prefix,
                    "generation": asdict(generation),
                    "reset_prefix_cache": reset_prefix_cache,
                }
            )
            if not self._connection.poll(self.inference_timeout):
                self.shutdown(force=True)
                raise TimeoutError(
                    f"{self.spec.label} 推理超过 {self.inference_timeout:g} 秒"
                )
            message = self._connection.recv()
            if message.get("kind") != "result":
                raise RuntimeError(message.get("message", "模型 worker 推理失败"))
            return message["result"]

    def shutdown(self, *, force: bool = False) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        self.ready_info = None
        if process is None:
            return
        if process.is_alive() and not force and connection is not None:
            try:
                connection.send({"type": "close"})
                if connection.poll(3.0):
                    connection.recv()
                process.join(timeout=3.0)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if connection is not None:
            connection.close()


class WorkerRegistry:
    """Own exactly one replaceable model worker for each visible UI slot."""

    def __init__(self) -> None:
        self._workers: dict[str, InferenceWorker] = {}
        self._lock = threading.Lock()

    def infer(
        self,
        slot_id: str,
        spec: ModelSpec,
        prefix: str,
        generation: GenerationSettings,
        *,
        reset_prefix_cache: bool,
    ) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.get(slot_id)
            if worker is None or worker.spec.runtime_key != spec.runtime_key:
                if worker is not None:
                    worker.shutdown()
                worker = InferenceWorker(spec)
                self._workers[slot_id] = worker
            return worker.infer(
                prefix,
                generation,
                reset_prefix_cache=reset_prefix_cache,
            )

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                slot_id: {
                    "loaded": worker.is_alive and worker.ready_info is not None,
                    "label": worker.spec.label,
                    "model_path": worker.spec.model_path,
                    "backend": worker.spec.backend,
                    "ready": worker.ready_info,
                }
                for slot_id, worker in self._workers.items()
            }

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.shutdown()


def _demo_suffixes(prefix: str, slot_id: str) -> list[str]:
    corpus = (
        (("报错", "版本", "代码"), ("先把完整日志贴出来。", "我再顺着调用链排查一下。", "先确认依赖版本有没有变化。")),
        (("小区", "门口", "到家"), ("你不用下来接我。", "我马上自己上去。", "我在楼下等你一会儿。")),
        (("周末", "下雨"), ("我们就去郊外走走。", "可以约着一起去爬山。", "咱们找个地方晒晒太阳。")),
        (("深度学习",), ("让神经网络从大量样本中自动学习规律。", "用多层神经网络逐步提取数据特征。", "通过反向传播不断调整网络参数。")),
        (("下班",), ("你们不用等我吃饭了。", "我到家以后再给你发消息。", "晚饭你先吃，不用特意等我。")),
    )
    for keywords, suffixes in corpus:
        if any(keyword in prefix for keyword in keywords):
            result = list(suffixes)
            break
    else:
        result = ["我晚点再认真看一下。", "等确认清楚以后告诉你。", "剩下的明天再接着处理。"]
    if slot_id == "b":
        result = [result[1], result[0], result[2]]
    return result


class DemoRegistry:
    """Deterministic no-GPU backend used for UI preview and CPU tests."""

    def __init__(self) -> None:
        self._request_count = {"a": 0, "b": 0}

    def infer(
        self,
        slot_id: str,
        spec: ModelSpec,
        prefix: str,
        generation: GenerationSettings,
        *,
        reset_prefix_cache: bool,
    ) -> dict[str, Any]:
        del reset_prefix_cache
        self._request_count[slot_id] += 1
        candidates = _demo_suffixes(prefix, slot_id)
        digest = hashlib.sha256(f"{prefix}:{slot_id}".encode()).digest()
        latency_ms = 82.0 + digest[0] / 8.0 + (28.0 if slot_id == "b" else 0.0)
        candidate_rows = [
            {
                "text": text,
                "token_count": 7 + index,
                "average_logprob": -0.18 - index * 0.07,
                "base_score": -0.18 - index * 0.07,
                "selection_score": -0.18 - index * 0.09,
                "stop_reason": "terminal_text",
                "invalid_reasons": [],
            }
            for index, text in enumerate(candidates)
        ]
        raw_candidates = [
            *candidate_rows,
            {**candidate_rows[0], "text": candidate_rows[0]["text"].rstrip("。")},
            {
                "text": "然后再",
                "token_count": 2,
                "average_logprob": -0.8,
                "base_score": None,
                "selection_score": None,
                "stop_reason": "max_new_tokens",
                "invalid_reasons": ["unfinished_fragment"],
            },
        ]
        active_tokens = 58 + digest[1] % 30
        return {
            "prefix": prefix,
            "generation_id": self._request_count[slot_id],
            "cancelled": False,
            "candidates": candidate_rows,
            "raw_candidates": raw_candidates,
            "prefix_tokens": max(2, len(prefix) // 2),
            "sampling_attempts": generation.sampling_attempts + 4,
            "generated_tokens": active_tokens,
            "active_model_tokens": active_tokens,
            "latency_ms": latency_ms,
            "gpu_latency_ms": latency_ms - 4.2,
            "unique_kv_pages": 28,
            "reused_prefix_tokens": 0,
            "refill_rounds": 1,
            "valid_unique_candidates": 3,
            "invalid_candidates": 1,
            "duplicate_candidates": 1,
            "refill_stop_reason": "filled",
            "runtime": {
                "label": spec.label,
                "model_path": spec.model_path,
                "configured_backend": spec.backend,
                "effective_backend": spec.backend,
                "model_load_ms": 0.0,
                "model_warmup_ms": 0.0,
                "api_wall_ms": latency_ms + 1.5,
                "active_tokens_per_second": active_tokens * 1000.0 / latency_ms,
                "gpu_active_tokens_per_second": active_tokens * 1000.0 / (latency_ms - 4.2),
                "cuda_allocated_mib": 468.0 if slot_id == "a" else 227.0,
                "cuda_reserved_mib": 512.0 if slot_id == "a" else 256.0,
                "cuda_peak_allocated_mib": 474.0 if slot_id == "a" else 231.0,
                "cold_request": False,
                "first_request": self._request_count[slot_id] == 1,
                "request_index": self._request_count[slot_id],
                "demo": True,
                "model": {
                    "model_type": "minimind_ime_v3",
                    "architecture_revision": "demo",
                    "residual_type": "block_attnres" if slot_id == "a" else "standard",
                    "attnres_backend": spec.backend,
                    "layers": 32 if slot_id == "a" else 14,
                    "hidden_size": 768,
                    "intermediate_size": 2048,
                    "vocab_size": 16384,
                    "context_length": 512,
                    "parameter_count": 214_063_360 if slot_id == "a" else 100_000_000,
                    "dtype": "bfloat16",
                },
            },
        }

    def health(self) -> dict[str, Any]:
        return {"demo": True, "requests": dict(self._request_count)}

    def shutdown(self) -> None:
        return None


def _normalized_candidate_set(result: dict[str, Any]) -> set[str]:
    return {
        str(item.get("text", "")).strip().rstrip("，。！？；：,.!?;:")
        for item in result.get("candidates", [])
        if str(item.get("text", "")).strip()
    }


def _character_lcp(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_candidates = left.get("candidates", [])
    right_candidates = right.get("candidates", [])
    left_texts = [str(item.get("text", "")) for item in left_candidates]
    right_texts = [str(item.get("text", "")) for item in right_candidates]
    overlap = _normalized_candidate_set(left) & _normalized_candidate_set(right)
    same_rank = sum(
        left_text.rstrip("。") == right_text.rstrip("。")
        for left_text, right_text in zip(left_texts, right_texts)
    )
    left_latency = float(left.get("latency_ms", 0.0))
    right_latency = float(right.get("latency_ms", 0.0))
    ratio = None
    if left_latency > 0.0 and right_latency > 0.0:
        ratio = max(left_latency, right_latency) / min(left_latency, right_latency)
    return {
        "top1_same": bool(left_texts and right_texts and left_texts[0] == right_texts[0]),
        "same_rank_candidates": same_rank,
        "top3_overlap": len(overlap),
        "overlap_candidates": sorted(overlap),
        "top1_character_lcp": _character_lcp(
            left_texts[0] if left_texts else "",
            right_texts[0] if right_texts else "",
        ),
        "latency_ratio": ratio,
        "faster_slot": (
            "a" if left_latency < right_latency else "b" if right_latency < left_latency else None
        ),
    }


class ImeCompareService:
    def __init__(
        self,
        default_slots: dict[str, ModelSpec],
        *,
        profiles: list[ModelSpec] | None = None,
        demo: bool = False,
    ) -> None:
        self.default_slots = default_slots
        self.profiles = profiles or list(default_slots.values())
        self.demo = demo
        self.registry: WorkerRegistry | DemoRegistry = DemoRegistry() if demo else WorkerRegistry()
        self._compare_lock = threading.Lock()

    def initial_config(self) -> dict[str, Any]:
        return {
            "demo": self.demo,
            "slots": {slot_id: asdict(spec) for slot_id, spec in self.default_slots.items()},
            "profiles": [asdict(profile) for profile in self.profiles],
            "generation": asdict(GenerationSettings()),
            "examples": list(DEFAULT_EXAMPLES),
            "order": "a_then_b",
        }

    def compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = CompareRequest.from_payload(
            payload,
            require_local_paths=not self.demo,
        )
        slot_order = ["a", "b"] if request.order == "a_then_b" else ["b", "a"]
        slot_order = [slot_id for slot_id in slot_order if slot_id in request.targets]
        results: dict[str, Any] = {}
        started = time.perf_counter()

        # One GPU should not execute both models concurrently: serial execution
        # makes the latency numbers comparable and avoids artificial OOM spikes.
        with self._compare_lock:
            for slot_id in slot_order:
                slot_started = time.perf_counter()
                try:
                    result = self.registry.infer(
                        slot_id,
                        request.slots[slot_id],
                        request.prefix,
                        request.generation,
                        reset_prefix_cache=request.reset_prefix_cache,
                    )
                    results[slot_id] = {
                        "ok": True,
                        "result": result,
                        "total_wall_ms": (time.perf_counter() - slot_started) * 1000.0,
                    }
                except Exception as error:
                    results[slot_id] = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "total_wall_ms": (time.perf_counter() - slot_started) * 1000.0,
                    }

        summary = None
        if results.get("a", {}).get("ok") and results.get("b", {}).get("ok"):
            summary = compare_results(results["a"]["result"], results["b"]["result"])
        return json_safe(
            {
                "prefix": request.prefix,
                "order": request.order,
                "results": results,
                "comparison": summary,
                "total_wall_ms": (time.perf_counter() - started) * 1000.0,
                "demo": self.demo,
            }
        )

    def health(self) -> dict[str, Any]:
        return {"ok": True, "demo": self.demo, "workers": self.registry.health()}

    def shutdown(self) -> None:
        self.registry.shutdown()
