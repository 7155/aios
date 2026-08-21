from __future__ import annotations

import os
import threading
import time

import pytest

from aios import ImeCompletionEngine, ImeGenerationConfig, LLM


MODEL_PATH = os.environ.get("AIOS_IME_MODEL")
pytestmark = pytest.mark.skipif(not MODEL_PATH, reason="set AIOS_IME_MODEL for GPU tests")


@pytest.fixture(scope="module")
def engine() -> ImeCompletionEngine:
    llm = LLM(
        MODEL_PATH,
        kv_cache_max_tokens=768,
        attention_workspace_size=8 * 2**20,
    )
    result = ImeCompletionEngine(llm)
    result.complete("预热", ImeGenerationConfig(seed=1))
    return result


def test_candidate_group_returns_three_and_recycles_branch_pages(
    engine: ImeCompletionEngine,
) -> None:
    result = engine.complete("没关系，你先忙你的，", ImeGenerationConfig(seed=7))
    assert len(result.candidates) == 3
    assert len({item.text for item in result.candidates}) == 3
    assert 8 <= result.sampling_attempts <= 24
    assert result.refill_stop_reason == "filled"
    assert engine.llm.num_pages - engine.llm.cache_manager.available_size == result.prefix_tokens


def test_incremental_prefix_reuses_token_lcp(engine: ImeCompletionEngine) -> None:
    engine.reset_prefix_cache()
    first = engine.complete("没关系，你先忙", ImeGenerationConfig(seed=11))
    extended = engine.complete("没关系，你先忙你的，", ImeGenerationConfig(seed=11))
    repeated = engine.complete("没关系，你先忙你的，", ImeGenerationConfig(seed=11))
    assert first.reused_prefix_tokens == 0
    assert extended.reused_prefix_tokens > 1
    assert repeated.reused_prefix_tokens == repeated.prefix_tokens


def test_stable_same_pinyin_scoring_avoids_decode_near_tie_flip(
    engine: ImeCompletionEngine,
) -> None:
    result = engine.score_candidates(
        "这本小说写尽了", ["世间", "时间", "实践", "事件"], mode="stable"
    )
    assert result.candidates[0].text == "世间"


def test_latest_generation_cancels_old_group_and_frees_pages(
    engine: ImeCompletionEngine,
) -> None:
    box: dict[str, object] = {}
    old_token = engine.new_generation()

    def run_old() -> None:
        box["result"] = engine.complete(
            "我正在连续输入一段比较长的内容，希望旧请求尽快停止",
            ImeGenerationConfig(
                max_new_tokens=64,
                min_new_tokens=64,
                max_candidate_chars=100,
                seed=17,
            ),
            old_token,
        )

    worker = threading.Thread(target=run_old)
    worker.start()
    time.sleep(0.015)
    new_token = engine.new_generation()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert box["result"].cancelled

    current = engine.complete("新的输入", ImeGenerationConfig(seed=19), new_token)
    assert not current.cancelled
    engine.reset_prefix_cache()
    assert engine.llm.cache_manager.available_size == engine.llm.num_pages
