from __future__ import annotations

import math

import torch

from aios.core import SamplingParams
from aios.engine.sample import Sampler, stateless_uniforms
from aios.ime import (
    ImeCandidate,
    ImeGenerationConfig,
    adaptive_refill_attempts,
    candidate_pool_stats,
    candidate_key,
    invalid_reasons,
    ngram_similarity,
    select_top_candidates,
    token_longest_common_prefix,
    truncate_at_terminal,
)


def candidate(text: str, score: float) -> ImeCandidate:
    return ImeCandidate(
        text=text,
        token_count=3,
        average_logprob=score,
        base_score=score,
        selection_score=score,
        stop_reason="eos",
        invalid_reasons=(),
    )


def invalid_candidate(text: str, reason: str = "unfinished_fragment") -> ImeCandidate:
    return ImeCandidate(
        text=text,
        token_count=2,
        average_logprob=-0.1,
        base_score=float("-inf"),
        selection_score=float("-inf"),
        stop_reason="max_new_tokens",
        invalid_reasons=(reason,),
    )


def test_candidate_key_collapses_display_punctuation() -> None:
    assert candidate_key(" 我晚点回复。 ") == candidate_key("我晚点回复")


def test_decoded_terminal_recovers_first_complete_sentence() -> None:
    text, stopped = truncate_at_terminal("我马上处理好。后面的句子不应显示")
    assert stopped
    assert text == "我马上处理好。"
    phrase, stopped = truncate_at_terminal("我先确认一下，再回复你")
    assert not stopped
    assert phrase == "我先确认一下，再回复你"


def test_invalid_reasons_reject_empty_assistant_and_repeat() -> None:
    config = ImeGenerationConfig()
    assert invalid_reasons("", config) == ("empty",)
    assert "assistant_template" in invalid_reasons("作为一个AI，我建议", config)
    assert "repeated_ngram" in invalid_reasons("好的好的好的", config)
    assert "unfinished_fragment" in invalid_reasons("我晚点再", config)
    assert "boundary_repeat" in invalid_reasons("好好休息", config, prefix="你好")


def test_boundary_repeat_allows_natural_word_boundary() -> None:
    config = ImeGenerationConfig()
    assert "boundary_repeat" not in invalid_reasons(
        "天气有点热，我先喝点水",
        config,
        prefix="今天",
    )


def test_mmr_selection_keeps_quality_and_diversity() -> None:
    items = [
        candidate("我晚点给你发消息", -0.10),
        candidate("我晚一点给你发消息", -0.11),
        candidate("等我回来再联系你", -0.16),
        candidate("处理完以后回复你", -0.18),
    ]
    selected = select_top_candidates(items, count=3, diversity_lambda=0.8)
    texts = [item.text for item in selected]
    assert texts[0] == "我晚点给你发消息"
    assert "等我回来再联系你" in texts
    assert "处理完以后回复你" in texts


def test_ngram_similarity_is_bounded() -> None:
    similarity = ngram_similarity("我晚点回复你", "我晚一点回复你")
    assert 0.0 < similarity < 1.0


def test_sampler_returns_raw_logprob_before_temperature() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    sampler = Sampler(SamplingParams(temperature=0.5, top_k=1, top_p=1.0))
    token, raw_logprob = sampler.sample_with_logprobs(logits)
    assert token.item() == 0
    expected = torch.log_softmax(logits, dim=-1)[0, 0].item()
    assert math.isclose(raw_logprob.item(), expected, rel_tol=1e-6)


def test_sampler_masks_end_tokens_without_changing_raw_logprob() -> None:
    logits = torch.tensor([[9.0, 8.0, 2.0, 1.0]])
    sampler = Sampler(SamplingParams(temperature=0.0))
    token, raw_logprob = sampler.sample_with_logprobs(
        logits,
        forbidden_token_ids=(0, 1),
    )
    assert token.item() == 2
    expected = torch.log_softmax(logits, dim=-1)[0, 2].item()
    assert math.isclose(raw_logprob.item(), expected, rel_tol=1e-6)


def test_sampler_supports_row_specific_repeat_bans() -> None:
    logits = torch.tensor([[9.0, 8.0, 1.0], [9.0, 8.0, 1.0]])
    sampler = Sampler(SamplingParams(temperature=0.0))
    token, raw_logprob = sampler.sample_with_logprobs(
        logits,
        forbidden_token_ids_by_row=torch.tensor([[0], [1]]),
    )
    assert token.squeeze(-1).tolist() == [1, 0]
    expected = torch.log_softmax(logits, dim=-1)
    assert math.isclose(raw_logprob[0].item(), expected[0, 1].item(), rel_tol=1e-6)
    assert math.isclose(raw_logprob[1].item(), expected[1, 0].item(), rel_tol=1e-6)


def test_candidate_random_stream_survives_active_row_compaction() -> None:
    device = torch.device("cpu")
    sampler = Sampler(SamplingParams(temperature=0.8, top_k=4, top_p=0.9))
    logits = torch.tensor([
        [3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    streams = stateless_uniforms([101, 202], steps=3, device=device)

    full, _ = sampler.sample_with_logprobs(logits, uniforms=streams[:, 1])
    compacted, _ = sampler.sample_with_logprobs(
        logits[1:],
        uniforms=streams[1:, 1],
    )
    assert compacted.item() == full[1].item()

    reversed_streams = stateless_uniforms([202, 101], steps=3, device=device)
    assert torch.equal(reversed_streams.flip(0), streams)


def test_token_longest_common_prefix_handles_retokenized_tail() -> None:
    assert token_longest_common_prefix([1, 2, 3, 4], [1, 2, 8]) == 2
    assert token_longest_common_prefix([1, 2], [1, 2, 3]) == 2


def test_generation_config_rejects_invalid_refill_settings() -> None:
    config = ImeGenerationConfig(sampling_attempts=8, max_sampling_attempts=7)
    try:
        config.validate()
    except ValueError as error:
        assert "max_sampling_attempts" in str(error)
    else:
        raise AssertionError("invalid adaptive sampling config was accepted")


def test_generation_config_rejects_invalid_minimum_length() -> None:
    config = ImeGenerationConfig(max_new_tokens=2, min_new_tokens=3)
    try:
        config.validate()
    except ValueError as error:
        assert "min_new_tokens" in str(error)
    else:
        raise AssertionError("invalid minimum generation length was accepted")


def test_candidate_pool_stats_count_invalid_and_display_duplicates() -> None:
    items = [
        candidate("我马上回去", -0.1),
        candidate("我马上回去。", -0.2),
        candidate("我晚点回去", -0.3),
        invalid_candidate("我等会再"),
    ]
    stats = candidate_pool_stats(items)
    assert stats.attempts == 4
    assert stats.valid_candidates == 3
    assert stats.valid_unique_candidates == 2
    assert stats.invalid_candidates == 1
    assert stats.duplicate_candidates == 1


def test_adaptive_refill_scales_with_filtered_unique_yield() -> None:
    config = ImeGenerationConfig(
        sampling_attempts=8,
        max_sampling_attempts=24,
        refill_batch_size=8,
    )
    duplicate_heavy = [candidate("我马上回去", -0.1)] * 4 + [
        invalid_candidate(f"坏候选{index}") for index in range(4)
    ]
    assert adaptive_refill_attempts(duplicate_heavy, config, 8) == 8

    two_unique = [
        candidate("我马上回去", -0.1),
        candidate("我晚点回去", -0.2),
        *[invalid_candidate(f"坏候选{index}") for index in range(6)],
    ]
    assert adaptive_refill_attempts(two_unique, config, 8) == 4

    full = [
        candidate("我马上回去", -0.1),
        candidate("我晚点回去", -0.2),
        candidate("我明天回去", -0.3),
    ]
    assert adaptive_refill_attempts(full, config, 8) == 0


def test_adaptive_refill_respects_remaining_attempt_budget() -> None:
    config = ImeGenerationConfig(max_sampling_attempts=10)
    items = [invalid_candidate(f"坏候选{index}") for index in range(8)]
    assert adaptive_refill_attempts(items, config, 8) == 2
