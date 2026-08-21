#!/usr/bin/env python3
"""Compare two AIOS Block AttnRes backends on a real exported model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="刚才同步的时候漏了一个细节，我先把漏掉的细节",
    )
    parser.add_argument(
        "--reference-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="eager",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=("reference", "eager", "compiled", "triton"),
        default="triton",
    )
    parser.add_argument("--kv-cache-max-tokens", type=int, default=512)
    parser.add_argument("--attention-workspace-mib", type=int, default=8)
    parser.add_argument(
        "--max-abs-tolerance",
        type=float,
        default=0.125,
        help="BF16 full-vocabulary logit tolerance after 65 depth mixers",
    )
    parser.add_argument(
        "--max-logprob-tolerance",
        type=float,
        default=1e-3,
        help="Maximum absolute difference between aligned candidate scores",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def set_attnres_backend(llm: LLM, backend: str) -> None:
    trunk = llm.model.model
    if llm.config.residual_type != "block_attnres":
        raise ValueError("the equivalence check requires a Block AttnRes model")
    for layer in trunk.layers.op_list:
        layer.attn_res_mixer._backend = backend
        layer.mlp_res_mixer._backend = backend
    trunk.final_attn_res_mixer._backend = backend


def prefix_logits(engine: ImeCompletionEngine, prefix: str) -> torch.Tensor:
    engine.reset_prefix_cache()
    token_ids = engine.llm.tokenizer.encode(prefix, add_special_tokens=False)
    bos_token_id = engine.llm.tokenizer.bos_token_id
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    if not token_ids:
        raise ValueError("prefix produced no input tokens")

    page_table = torch.zeros(
        (1, len(token_ids)), dtype=torch.int32, device=engine.llm.device
    )
    engine.llm.ctx.page_table = page_table
    logits, _, _ = engine._prepare_prefix(token_ids, page_table)
    result = logits.detach().clone()
    engine.reset_prefix_cache()
    return result


def topk_overlap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    count: int,
) -> int:
    reference_ids = set(reference.topk(count).indices.cpu().tolist())
    candidate_ids = set(candidate.topk(count).indices.cpu().tolist())
    return len(reference_ids & candidate_ids)


def completion_signature(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "text": item.text,
            "token_count": item.token_count,
            "average_logprob": item.average_logprob,
            "stop_reason": item.stop_reason,
            "invalid_reasons": list(item.invalid_reasons),
        }
        for item in result.raw_candidates
    ]


def discrete_completion_signature(result: Any) -> list[dict[str, Any]]:
    """Keep only fields that define the user-visible sampled sequence."""
    return [
        {
            "text": item.text,
            "token_count": item.token_count,
            "stop_reason": item.stop_reason,
            "invalid_reasons": list(item.invalid_reasons),
        }
        for item in result.raw_candidates
    ]


def render_markdown(report: dict[str, Any]) -> str:
    logits = report["prefix_logits"]
    completion = report["fixed8_completion"]
    return f"""# Block AttnRes 真实模型运行时等价性

- 模型：`{report['model']}`
- 架构：`{report['architecture_revision']}`
- 对比：`{report['reference_backend']}` → `{report['candidate_backend']}`
- Prefix logits max/mean absolute diff：`{logits['max_abs_diff']:.8g}` / `{logits['mean_abs_diff']:.8g}`
- Cosine similarity：`{logits['cosine_similarity']:.10f}`
- Top-1/3/10/50 overlap：`{logits['topk_overlap']['1']}/1`、`{logits['topk_overlap']['3']}/3`、`{logits['topk_overlap']['10']}/10`、`{logits['topk_overlap']['50']}/50`
- 固定 8 路候选文本/停止状态完全一致：`{completion['discrete_exact_match']}`
- 候选 average logprob 最大差值：`{completion['max_average_logprob_diff']:.8g}`

该检查加载同一份导出权重，在相同裸中文 Prefix、KV 配置、采样参数和 seed 下分别执行
Direct/Eager 与优化后端。它同时比较 Prefix 最后位置的完整词表 logits，以及固定 8 路生成
得到的原始候选文本、token 数、停止原因和 logprob。
"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    llm = LLM(
        str(args.model),
        attnres_backend=args.reference_backend,
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        attention_workspace_size=args.attention_workspace_mib * 2**20,
    )
    engine = ImeCompletionEngine(llm)

    set_attnres_backend(llm, args.reference_backend)
    reference_logits = prefix_logits(engine, args.prefix).float().flatten()
    set_attnres_backend(llm, args.candidate_backend)
    candidate_logits = prefix_logits(engine, args.prefix).float().flatten()
    torch.cuda.synchronize()

    absolute_difference = (reference_logits - candidate_logits).abs()
    logits_report = {
        "max_abs_diff": float(absolute_difference.max().item()),
        "mean_abs_diff": float(absolute_difference.mean().item()),
        "cosine_similarity": float(
            F.cosine_similarity(
                reference_logits.unsqueeze(0), candidate_logits.unsqueeze(0)
            ).item()
        ),
        "argmax_equal": bool(
            reference_logits.argmax().item() == candidate_logits.argmax().item()
        ),
        "topk_overlap": {
            str(count): topk_overlap(reference_logits, candidate_logits, count)
            for count in (1, 3, 10, 50)
        },
    }

    generation_config = ImeGenerationConfig(
        sampling_attempts=8,
        max_sampling_attempts=8,
        refill_batch_size=8,
        max_new_tokens=12,
        max_candidate_chars=32,
        seed=20260821,
    )
    set_attnres_backend(llm, args.reference_backend)
    engine.reset_prefix_cache()
    reference_completion = engine.complete(args.prefix, generation_config)
    set_attnres_backend(llm, args.candidate_backend)
    engine.reset_prefix_cache()
    candidate_completion = engine.complete(args.prefix, generation_config)
    reference_signature = completion_signature(reference_completion)
    candidate_signature = completion_signature(candidate_completion)
    reference_discrete = discrete_completion_signature(reference_completion)
    candidate_discrete = discrete_completion_signature(candidate_completion)
    average_logprob_differences = [
        abs(
            reference_item["average_logprob"]
            - candidate_item["average_logprob"]
        )
        for reference_item, candidate_item in zip(
            reference_signature, candidate_signature, strict=True
        )
    ]

    report = {
        "schema_version": "aios.attnres_runtime_equivalence.v1",
        "model": str(args.model.resolve()),
        "architecture_revision": llm.config.architecture_revision,
        "reference_backend": args.reference_backend,
        "candidate_backend": args.candidate_backend,
        "prefix": args.prefix,
        "dtype": str(llm.dtype),
        "prefix_logits": logits_report,
        "fixed8_completion": {
            "discrete_exact_match": reference_discrete == candidate_discrete,
            "max_average_logprob_diff": max(
                average_logprob_differences, default=0.0
            ),
            "reference": reference_signature,
            "candidate": candidate_signature,
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    markdown = render_markdown(report)
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if logits_report["max_abs_diff"] > args.max_abs_tolerance:
        raise RuntimeError(
            "AttnRes backend logit difference exceeds tolerance: "
            f"{logits_report['max_abs_diff']} > {args.max_abs_tolerance}"
        )
    if not report["fixed8_completion"]["discrete_exact_match"]:
        raise RuntimeError(
            "fixed-8 candidate text or stop state differs between backends"
        )
    if (
        report["fixed8_completion"]["max_average_logprob_diff"]
        > args.max_logprob_tolerance
    ):
        raise RuntimeError(
            "candidate logprob difference exceeds tolerance: "
            f"{report['fixed8_completion']['max_average_logprob_diff']} > "
            f"{args.max_logprob_tolerance}"
        )


if __name__ == "__main__":
    main()
