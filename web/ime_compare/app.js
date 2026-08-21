"use strict";

const FEEDBACK_KEY = "aios-ime-compare-feedback-v1";

const state = {
  config: null,
  busy: false,
  lastResponse: null,
  runId: null,
  toastTimer: null,
};

const $ = (id) => document.getElementById(id);

const reasonLabels = {
  empty: "空候选",
  assistant_template: "助手腔",
  repeated_ngram: "重复片段",
  unfinished_fragment: "未完成句式",
  boundary_repeat: "前缀边界复读",
  invalid_character: "非法字符",
  too_long: "超出显示长度",
  duplicate: "重复候选",
};

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function compactNumber(value) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("zh-CN", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 2400);
}

function setRuntimeState(kind, text) {
  const root = $("runtime-state");
  root.classList.remove("ready", "error");
  if (kind) root.classList.add(kind);
  $("runtime-state-text").textContent = text;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`本地服务返回了无法解析的响应（HTTP ${response.status}）`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

function setInputValue(id, value) {
  const input = $(id);
  if (input && value !== undefined && value !== null) {
    input.value = String(value);
  }
}

function profileMatchesSlot(profile, slot) {
  return profile.model_path === slot.model_path && profile.backend === slot.backend;
}

function applyProfile(slot, profileIndex) {
  if (profileIndex === "custom") return;
  const profile = state.config.profiles[Number(profileIndex)];
  if (!profile) return;
  setInputValue(`label-${slot}`, profile.label);
  setInputValue(`model-path-${slot}`, profile.model_path);
  setInputValue(`backend-${slot}`, profile.backend);
  $(`result-label-${slot}`).textContent = profile.label;
}

function hydrateProfiles(slot, selectedSlot) {
  const select = $(`profile-${slot}`);
  select.replaceChildren();
  state.config.profiles.forEach((profile, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${profile.label} · BF16`;
    select.append(option);
  });
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = "自定义本地目录";
  select.append(custom);
  const selectedIndex = state.config.profiles.findIndex((profile) =>
    profileMatchesSlot(profile, selectedSlot),
  );
  select.value = selectedIndex >= 0 ? String(selectedIndex) : "custom";
  select.addEventListener("change", () => applyProfile(slot, select.value));
  for (const id of [`label-${slot}`, `model-path-${slot}`, `backend-${slot}`]) {
    $(id).addEventListener("input", () => {
      const matchingIndex = state.config.profiles.findIndex(
        (profile) =>
          profile.model_path === $(`model-path-${slot}`).value.trim() &&
          profile.backend === $(`backend-${slot}`).value,
      );
      select.value = matchingIndex >= 0 ? String(matchingIndex) : "custom";
    });
  }
}

function hydrateConfig(config) {
  state.config = config;
  state.config.profiles = config.profiles || Object.values(config.slots);
  for (const slot of ["a", "b"]) {
    const model = config.slots[slot];
    setInputValue(`label-${slot}`, model.label);
    setInputValue(`model-path-${slot}`, model.model_path);
    setInputValue(`backend-${slot}`, model.backend);
    $(`result-label-${slot}`).textContent = model.label;
    hydrateProfiles(slot, model);
  }

  const generationIds = {
    "sampling-attempts": "sampling_attempts",
    "max-sampling-attempts": "max_sampling_attempts",
    "refill-batch-size": "refill_batch_size",
    "max-new-tokens": "max_new_tokens",
    temperature: "temperature",
    "top-k": "top_k",
    "top-p": "top_p",
    seed: "seed",
    "diversity-lambda": "diversity_lambda",
  };
  Object.entries(generationIds).forEach(([id, key]) => {
    setInputValue(id, config.generation[key]);
  });
  setInputValue("run-order", config.order);

  const examples = $("example-list");
  examples.replaceChildren();
  config.examples.forEach((prefix) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = prefix;
    button.addEventListener("click", () => {
      $("prefix-input").value = prefix;
      updateCharacterCount();
      $("prefix-input").focus();
    });
    examples.append(button);
  });

  $("demo-banner").hidden = !config.demo;
  setRuntimeState("ready", config.demo ? "Demo 服务已连接" : "本地推理服务已连接");
}

function updateCharacterCount() {
  $("char-count").textContent = String($("prefix-input").value.length);
}

function readNumber(id) {
  const value = $(id).value;
  if (value === "") throw new Error(`${id} 不能为空`);
  return Number(value);
}

function buildSlot(slot) {
  const original = state.config.slots[slot];
  return {
    ...original,
    label: $(`label-${slot}`).value.trim(),
    model_path: $(`model-path-${slot}`).value.trim(),
    backend: $(`backend-${slot}`).value,
  };
}

function buildPayload(targets) {
  const prefix = $("prefix-input").value;
  if (!prefix.trim()) throw new Error("先输入一个中文前缀");
  const generation = {
    ...state.config.generation,
    sampling_attempts: readNumber("sampling-attempts"),
    max_sampling_attempts: readNumber("max-sampling-attempts"),
    refill_batch_size: readNumber("refill-batch-size"),
    max_new_tokens: readNumber("max-new-tokens"),
    temperature: readNumber("temperature"),
    top_k: readNumber("top-k"),
    top_p: readNumber("top-p"),
    seed: readNumber("seed"),
    diversity_lambda: readNumber("diversity-lambda"),
  };
  return {
    prefix,
    slots: { a: buildSlot("a"), b: buildSlot("b") },
    generation,
    targets,
    order: $("run-order").value,
    reset_prefix_cache: $("reset-prefix-cache").checked,
  };
}

function setBusy(busy, targets = ["a", "b"]) {
  state.busy = busy;
  ["compare-button", "run-a-button", "run-b-button", "unload-button"].forEach((id) => {
    $(id).disabled = busy;
  });
  if (busy) {
    $("form-error").hidden = true;
    for (const slot of targets) {
      const status = $(`status-${slot}`);
      status.className = "run-status loading";
      status.textContent = "运行中";
      const candidates = $(`candidates-${slot}`);
      candidates.replaceChildren();
      const loading = document.createElement("div");
      loading.className = "loading-state";
      loading.textContent = "首次使用该配置时会先加载模型并编译 CUDA kernel";
      candidates.append(loading);
      $(`diagnostics-${slot}`).hidden = true;
      $(`raw-panel-${slot}`).hidden = true;
    }
  }
}

function setStatus(slot, kind, text) {
  const status = $(`status-${slot}`);
  status.className = `run-status ${kind}`;
  status.textContent = text;
}

function candidateRow(slot, rank, candidate, prefix) {
  const row = document.createElement("div");
  row.className = "candidate-row";

  const rankElement = document.createElement("span");
  rankElement.className = "candidate-rank";
  rankElement.textContent = `0${rank}`;

  const content = document.createElement("div");
  const text = document.createElement("p");
  text.className = "candidate-text";
  text.textContent = candidate.text;
  const score = document.createElement("p");
  score.className = "candidate-score";
  score.textContent = `avg logprob ${formatNumber(candidate.average_logprob, 4)} · ${candidate.token_count} tokens`;
  content.append(text, score);

  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-candidate";
  copy.textContent = "复制整句";
  copy.setAttribute("aria-label", `复制模型 ${slot.toUpperCase()} 第 ${rank} 条完整句子`);
  copy.addEventListener("click", async () => {
    const completeText = `${prefix}${candidate.text}`;
    try {
      await navigator.clipboard.writeText(completeText);
      showToast(`已复制：${completeText}`);
    } catch (_error) {
      showToast("浏览器未授权剪贴板，请手动复制");
    }
  });

  row.append(rankElement, content, copy);
  return row;
}

function metric(value, label) {
  const root = document.createElement("div");
  root.className = "metric";
  const strong = document.createElement("strong");
  strong.textContent = value;
  const span = document.createElement("span");
  span.textContent = label;
  root.append(strong, span);
  return root;
}

function renderMetrics(slot, result) {
  const root = $(`metrics-${slot}`);
  const runtime = result.runtime || {};
  root.replaceChildren(
    metric(`${formatNumber(result.latency_ms)} ms`, "完整 Top-3 延迟"),
    metric(`${formatNumber(result.gpu_latency_ms)} ms`, "GPU 事件延迟"),
    metric(`${formatNumber(runtime.active_tokens_per_second, 0)}`, "ACTIVE TOKENS / S"),
    metric(String(result.sampling_attempts ?? "—"), "实际采样路数"),
    metric(String(result.refill_rounds ?? "—"), "补采样轮次"),
    metric(`${formatNumber(runtime.cuda_peak_allocated_mib)} MiB`, "峰值显存"),
    metric(String(result.valid_unique_candidates ?? "—"), "有效互异候选"),
    metric(String(result.reused_prefix_tokens ?? "—"), "复用前缀 TOKENS"),
    metric(compactNumber(runtime.model?.parameter_count), "主模型参数"),
  );
  $(`diagnostics-${slot}`).hidden = false;
}

function rawStatus(candidate) {
  const reasons = candidate.invalid_reasons || [];
  if (!reasons.length) return "有效";
  return reasons.map((reason) => reasonLabels[reason] || reason).join("、");
}

function renderRawCandidates(slot, candidates) {
  const body = $(`raw-${slot}`);
  body.replaceChildren();
  candidates.forEach((candidate, index) => {
    const row = document.createElement("tr");
    const rank = document.createElement("td");
    rank.textContent = String(index + 1);
    const text = document.createElement("td");
    text.textContent = candidate.text || "（空）";
    const logprob = document.createElement("td");
    logprob.textContent = formatNumber(candidate.average_logprob, 4);
    const status = document.createElement("td");
    status.textContent = rawStatus(candidate);
    if ((candidate.invalid_reasons || []).length) status.className = "raw-invalid";
    row.append(rank, text, logprob, status);
    body.append(row);
  });
  $(`raw-panel-${slot}`).hidden = candidates.length === 0;
}

function architectureLabel(runtime) {
  const model = runtime.model || {};
  const residual = model.residual_type === "block_attnres" ? "Block AttnRes" : "Standard";
  const backend = runtime.effective_backend || runtime.configured_backend || "default";
  const layers = model.layers ? `${model.layers}L` : "?L";
  const dtype = model.dtype === "bfloat16" ? "BF16" : String(model.dtype || "").toUpperCase();
  return model.residual_type === "block_attnres"
    ? `${layers} · ${residual} · ${backend} · ${dtype}`
    : `${layers} · ${residual} residual · ${dtype}`;
}

function renderSlot(slot, envelope, prefix) {
  if (!envelope || !envelope.ok) {
    const error = envelope?.error || "该侧没有返回结果";
    setStatus(slot, "error", "失败");
    const candidates = $(`candidates-${slot}`);
    candidates.replaceChildren();
    const errorNode = document.createElement("div");
    errorNode.className = "error-state";
    errorNode.textContent = error;
    candidates.append(errorNode);
    $(`diagnostics-${slot}`).hidden = true;
    $(`raw-panel-${slot}`).hidden = true;
    return;
  }

  const result = envelope.result;
  const runtime = result.runtime || {};
  $(`result-label-${slot}`).textContent = runtime.label || $(`label-${slot}`).value;
  $(`result-meta-${slot}`).textContent = architectureLabel(runtime);
  setStatus(slot, "success", runtime.cold_request ? "完成 · 冷启动" : "完成");
  const candidates = $(`candidates-${slot}`);
  candidates.replaceChildren();
  if (!result.candidates?.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "没有通过过滤的候选";
    candidates.append(empty);
  } else {
    result.candidates.forEach((candidate, index) => {
      candidates.append(candidateRow(slot, index + 1, candidate, prefix));
    });
  }
  renderMetrics(slot, result);
  renderRawCandidates(slot, result.raw_candidates || []);
}

function renderComparison(comparison) {
  const strip = $("comparison-strip");
  if (!comparison) {
    strip.hidden = true;
    return;
  }
  $("overlap-value").textContent = `${comparison.top3_overlap} / 3`;
  $("rank-value").textContent = `${comparison.same_rank_candidates} / 3`;
  $("lcp-value").textContent = `${comparison.top1_character_lcp} 字`;
  if (comparison.latency_ratio && comparison.faster_slot) {
    $("latency-value").textContent = `${comparison.faster_slot.toUpperCase()} 快 ${formatNumber(comparison.latency_ratio, 2)}×`;
  } else {
    $("latency-value").textContent = "持平";
  }
  strip.hidden = false;
}

function showFormError(message) {
  const error = $("form-error");
  error.textContent = message;
  error.hidden = false;
}

async function runComparison(targets) {
  if (state.busy) return;
  let payload;
  try {
    payload = buildPayload(targets);
  } catch (error) {
    showFormError(error.message);
    return;
  }

  setBusy(true, targets);
  const buttonText = $("compare-button").querySelector("span");
  const originalButtonText = buttonText.textContent;
  buttonText.textContent = targets.length === 2 ? "正在串行运行 A / B…" : `正在运行 ${targets[0].toUpperCase()}…`;
  try {
    const response = await fetchJson("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.lastResponse = response;
    state.runId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    for (const slot of targets) {
      renderSlot(slot, response.results[slot], response.prefix);
    }
    renderComparison(response.comparison);
    $("result-timestamp").textContent = `${new Date().toLocaleTimeString("zh-CN", { hour12: false })} · 总墙钟 ${formatNumber(response.total_wall_ms)} ms`;
    const bothSuccessful = response.results.a?.ok && response.results.b?.ok;
    $("feedback-panel").hidden = !bothSuccessful;
    document.querySelectorAll("[data-preference]").forEach((button) => button.classList.remove("saved"));
    updateFeedbackCount();
    $("result-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showFormError(error.message);
    targets.forEach((slot) => renderSlot(slot, { ok: false, error: error.message }, payload.prefix));
  } finally {
    setBusy(false);
    buttonText.textContent = originalButtonText;
  }
}

function loadFeedback() {
  try {
    const parsed = JSON.parse(localStorage.getItem(FEEDBACK_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}

function updateFeedbackCount() {
  $("feedback-count").textContent = `已记录 ${loadFeedback().length} 条`;
}

function recordFeedback(preference) {
  if (!state.lastResponse || !state.lastResponse.results.a?.ok || !state.lastResponse.results.b?.ok) {
    showToast("请先完成一次 A / B 对比");
    return;
  }
  const records = loadFeedback().filter((item) => item.run_id !== state.runId);
  const response = state.lastResponse;
  const record = {
    schema_version: "aios.ime_compare.feedback.v1",
    run_id: state.runId,
    created_at: new Date().toISOString(),
    prefix: response.prefix,
    preference,
    order: response.order,
    models: {
      a: response.results.a.result.runtime,
      b: response.results.b.result.runtime,
    },
    candidates: {
      a: response.results.a.result.candidates,
      b: response.results.b.result.candidates,
    },
    metrics: {
      a: {
        latency_ms: response.results.a.result.latency_ms,
        gpu_latency_ms: response.results.a.result.gpu_latency_ms,
        sampling_attempts: response.results.a.result.sampling_attempts,
      },
      b: {
        latency_ms: response.results.b.result.latency_ms,
        gpu_latency_ms: response.results.b.result.gpu_latency_ms,
        sampling_attempts: response.results.b.result.sampling_attempts,
      },
      comparison: response.comparison,
    },
  };
  records.push(record);
  localStorage.setItem(FEEDBACK_KEY, JSON.stringify(records.slice(-1000)));
  document.querySelectorAll("[data-preference]").forEach((button) => {
    button.classList.toggle("saved", button.dataset.preference === preference);
  });
  updateFeedbackCount();
  showToast("本次判断已保存在浏览器");
}

function exportFeedback() {
  const records = loadFeedback();
  if (!records.length) {
    showToast("当前没有可导出的对比记录");
    return;
  }
  const body = `${records.map((item) => JSON.stringify(item)).join("\n")}\n`;
  const blob = new Blob([body], { type: "application/x-ndjson;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `aios-ime-feedback-${new Date().toISOString().slice(0, 10)}.jsonl`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function unloadWorkers() {
  if (state.busy) return;
  $("unload-button").disabled = true;
  try {
    await fetchJson("/api/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    setStatus("a", "idle", "已卸载");
    setStatus("b", "idle", "已卸载");
    setRuntimeState("ready", "模型显存已释放");
    showToast("两个模型 worker 已关闭，GPU 显存已释放");
  } catch (error) {
    showFormError(error.message);
  } finally {
    $("unload-button").disabled = false;
  }
}

async function bootstrap() {
  try {
    const config = await fetchJson("/api/config");
    hydrateConfig(config);
    updateCharacterCount();
    updateFeedbackCount();
    $("prefix-input").focus();
  } catch (error) {
    setRuntimeState("error", "本地服务连接失败");
    showFormError(error.message);
  }
}

$("compare-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runComparison(["a", "b"]);
});
$("run-a-button").addEventListener("click", () => runComparison(["a"]));
$("run-b-button").addEventListener("click", () => runComparison(["b"]));
$("unload-button").addEventListener("click", unloadWorkers);
$("prefix-input").addEventListener("input", updateCharacterCount);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    runComparison(["a", "b"]);
  }
});
document.querySelectorAll("[data-preference]").forEach((button) => {
  button.addEventListener("click", () => recordFeedback(button.dataset.preference));
});
$("export-feedback").addEventListener("click", exportFeedback);
$("clear-feedback").addEventListener("click", () => {
  if (window.confirm("确认清空当前浏览器中的全部对比记录吗？")) {
    localStorage.removeItem(FEEDBACK_KEY);
    updateFeedbackCount();
    showToast("本地对比记录已清空");
  }
});

bootstrap();
