"use strict";

import {
  decodedFragment,
  escapeHtml,
  fetchJson,
  fetchText,
  isEditableTarget,
  normalizePath,
  requestedDocument,
  scrollToFragment,
} from "./course-utils.js";
import {
  buildOutline,
  configureRenderers,
  renderMarkdown,
  showDocumentError,
  showFatalError,
} from "./course-render.js";

const elements = {};
const state = {
  manifest: null,
  allDocuments: [],
  documentByPath: new Map(),
  currentPath: "",
  currentDocument: null,
  activeFilter: "all",
  searchQuery: "",
  completed: new Set(),
};

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  cacheElements();
  try {
    state.manifest = await fetchJson("site-manifest.json");
    state.allDocuments = state.manifest.sections.flatMap((section) =>
      section.docs.map((doc) => ({
        ...doc,
        sectionId: section.id,
        track: section.track,
      })),
    );
    state.documentByPath = new Map(
      state.allDocuments.map((doc) => [normalizePath(doc.path), doc]),
    );
    state.completed = loadProgress();

    configureRenderers();
    bindEvents();
    buildNavigation();
    updateProgressSummary();

    await loadDocument(requestedDocument() || state.manifest.course.home, {
      updateHistory: false,
      fragment: decodedFragment(),
    });
  } catch (error) {
    showFatalError(elements.content, error);
  }
}

function cacheElements() {
  const byId = (id) => document.getElementById(id);
  Object.assign(elements, {
    sidebar: byId("course-sidebar"),
    backdrop: byId("sidebar-backdrop"),
    menuButton: byId("menu-button"),
    searchButton: byId("search-button"),
    navigation: byId("course-navigation"),
    search: byId("course-search"),
    filterButtons: [...document.querySelectorAll("[data-filter]")],
    content: byId("lesson-content"),
    kicker: byId("document-kicker"),
    meta: byId("document-meta"),
    completeButton: byId("complete-button"),
    progressText: byId("progress-text"),
    progressBar: byId("progress-bar"),
    outline: byId("outline-navigation"),
    pager: byId("document-pager"),
  });
}

function bindEvents() {
  elements.menuButton.addEventListener("click", () => {
    setSidebarOpen(!elements.sidebar.classList.contains("is-open"));
  });
  elements.searchButton.addEventListener("click", () => {
    setSearchOpen(!elements.search.parentElement.classList.contains("is-open"));
  });
  elements.backdrop.addEventListener("click", () => setSidebarOpen(false));
  elements.search.addEventListener("input", (event) => {
    state.searchQuery = event.target.value.trim().toLocaleLowerCase("zh-CN");
    applyNavigationFilters();
  });
  elements.filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeFilter = button.dataset.filter || "all";
      elements.filterButtons.forEach((item) =>
        item.classList.toggle("is-active", item === button),
      );
      applyNavigationFilters();
    });
  });
  elements.completeButton.addEventListener("click", toggleCurrentCompletion);
  document.querySelectorAll("[data-doc-link]").forEach((anchor) => {
    anchor.addEventListener("click", (event) => {
      event.preventDefault();
      loadDocument(anchor.dataset.docLink);
    });
  });
  window.addEventListener("popstate", () => {
    loadDocument(requestedDocument() || state.manifest.course.home, {
      updateHistory: false,
      fragment: decodedFragment(),
    });
  });
  document.addEventListener("keydown", handleKeyboard);
}

function handleKeyboard(event) {
  if (
    event.key === "/" &&
    !event.metaKey &&
    !event.ctrlKey &&
    !isEditableTarget(event.target)
  ) {
    event.preventDefault();
    setSearchOpen(true);
    elements.search.focus();
    return;
  }
  if (event.key === "Escape") {
    setSidebarOpen(false);
    setSearchOpen(false);
    elements.search.blur();
    return;
  }
  if (isEditableTarget(event.target) || event.metaKey || event.ctrlKey) return;
  if (event.key === "[") navigateRelative(-1);
  if (event.key === "]") navigateRelative(1);
}

function buildNavigation() {
  elements.navigation.replaceChildren();
  for (const section of state.manifest.sections) {
    const sectionElement = document.createElement("section");
    sectionElement.className = "navigation-section";
    sectionElement.dataset.track = section.track;

    const heading = document.createElement("p");
    heading.className = "navigation-section-title";
    heading.innerHTML = `<span>${escapeHtml(section.title)}</span><span>${section.docs.length}</span>`;

    const list = document.createElement("ul");
    list.className = "navigation-list";
    for (const doc of section.docs) {
      const item = document.createElement("li");
      item.className = "navigation-item";
      item.dataset.path = normalizePath(doc.path);
      item.dataset.search = [doc.id, doc.title, doc.summary || "", ...(doc.concepts || [])]
        .join(" ")
        .toLocaleLowerCase("zh-CN");

      const button = document.createElement("button");
      button.className = "navigation-link";
      button.type = "button";
      button.dataset.path = normalizePath(doc.path);
      button.innerHTML = `
        <span class="navigation-id">${escapeHtml(doc.id)}</span>
        <span class="navigation-title">${escapeHtml(doc.title)}</span>
        <span class="navigation-check" aria-label="完成状态">${
          doc.kind === "lesson" ? "○" : "·"
        }</span>
      `;
      button.addEventListener("click", () => loadDocument(doc.path));
      item.append(button);
      list.append(item);
    }
    sectionElement.append(heading, list);
    elements.navigation.append(sectionElement);
  }
  updateNavigationCompletion();
  applyNavigationFilters();
}

function applyNavigationFilters() {
  let visibleItems = 0;
  for (const section of elements.navigation.querySelectorAll(".navigation-section")) {
    const trackMatches =
      state.activeFilter === "all" || section.dataset.track === state.activeFilter;
    let visibleInSection = 0;
    for (const item of section.querySelectorAll(".navigation-item")) {
      const queryMatches =
        !state.searchQuery || item.dataset.search.includes(state.searchQuery);
      item.hidden = !(trackMatches && queryMatches);
      if (!item.hidden) visibleInSection += 1;
    }
    section.hidden = visibleInSection === 0;
    visibleItems += visibleInSection;
  }

  elements.navigation.querySelector(".navigation-empty")?.remove();
  if (!visibleItems) {
    const empty = document.createElement("p");
    empty.className = "navigation-empty";
    empty.textContent = "没有匹配课程。请缩短关键词，或切回“全部”。";
    elements.navigation.append(empty);
  }
}

async function loadDocument(path, options = {}) {
  const normalized = normalizePath(path);
  const doc = state.documentByPath.get(normalized);
  if (!doc) {
    showDocumentFailure(`课程清单中不存在“${normalized}”。请从左侧目录重新进入。`);
    return;
  }

  state.currentPath = normalized;
  state.currentDocument = doc;
  setLoadingState(doc);

  try {
    const markdown = await fetchText(normalized);
    await renderMarkdown({
      markdown,
      currentPath: normalized,
      content: elements.content,
      documentByPath: state.documentByPath,
      onNavigate: (nextPath, fragment) => loadDocument(nextPath, { fragment }),
    });
    buildOutline(elements.content, elements.outline);
    buildPager();
    updateCurrentNavigation();
    updateCompleteButton();
    updateDocumentMeta(doc);

    const fragment = options.fragment ?? (options.updateHistory === false ? decodedFragment() : "");
    if (options.updateHistory !== false) {
      const url = new URL(location.href);
      url.searchParams.set("doc", normalized);
      url.hash = fragment ? encodeURIComponent(fragment) : "";
      history.pushState({}, "", url);
    }

    document.title = `${doc.id} · ${doc.title} | AIOS-IME`;
    elements.content.focus({ preventScroll: true });
    requestAnimationFrame(() => {
      if (fragment) scrollToFragment(fragment);
      else window.scrollTo({ top: 0, behavior: "auto" });
    });
    setSidebarOpen(false);
    setSearchOpen(false);
  } catch (error) {
    showDocumentFailure(`${doc.id} · ${doc.title}\n\n${errorMessage(error)}`);
  }
}

function setLoadingState(doc) {
  elements.kicker.textContent = `${doc.sectionId} · ${doc.id}`;
  elements.meta.textContent = "加载教材中……";
  elements.content.innerHTML = `
    <div class="loading-card" aria-label="加载中">
      <div class="loading-line loading-line-wide"></div>
      <div class="loading-line"></div>
      <div class="loading-line"></div>
    </div>
  `;
  elements.outline.replaceChildren();
  elements.pager.replaceChildren();
}

function showDocumentFailure(message) {
  showDocumentError(elements.content, message);
  elements.meta.textContent = "教材加载失败";
  elements.completeButton.hidden = true;
  elements.outline.replaceChildren();
  elements.pager.replaceChildren();
}

function buildPager() {
  elements.pager.replaceChildren();
  const index = state.allDocuments.findIndex(
    (doc) => normalizePath(doc.path) === state.currentPath,
  );
  if (index < 0) return;
  const previous = state.allDocuments[index - 1];
  const next = state.allDocuments[index + 1];
  elements.pager.append(
    previous ? createPagerLink(previous, "上一篇", false) : document.createElement("span"),
    next ? createPagerLink(next, "下一篇", true) : document.createElement("span"),
  );
}

function createPagerLink(doc, label, isNext) {
  const anchor = document.createElement("a");
  anchor.href = `?doc=${encodeURIComponent(doc.path)}`;
  anchor.className = `pager-link${isNext ? " is-next" : ""}`;
  anchor.innerHTML = `
    <span class="pager-label">${label}</span>
    <span class="pager-title">${escapeHtml(doc.id)} · ${escapeHtml(doc.title)}</span>
  `;
  anchor.addEventListener("click", (event) => {
    event.preventDefault();
    loadDocument(doc.path);
  });
  return anchor;
}

function updateDocumentMeta(doc) {
  const parts = [];
  if (doc.duration) parts.push(doc.duration);
  if (doc.output) parts.push(`产物：${doc.output}`);
  if (doc.concepts?.length) parts.push(doc.concepts.join(" · "));
  elements.kicker.textContent = `${doc.sectionId} · ${doc.id}`;
  elements.meta.textContent = parts.join("　") || doc.summary || "";
}

function updateCurrentNavigation() {
  for (const link of elements.navigation.querySelectorAll(".navigation-link")) {
    const current = normalizePath(link.dataset.path) === state.currentPath;
    link.classList.toggle("is-current", current);
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function toggleCurrentCompletion() {
  if (state.currentDocument?.kind !== "lesson") return;
  const id = state.currentDocument.id;
  if (state.completed.has(id)) state.completed.delete(id);
  else state.completed.add(id);
  persistProgress();
  updateCompleteButton();
  updateNavigationCompletion();
  updateProgressSummary();
}

function updateCompleteButton() {
  const doc = state.currentDocument;
  if (!doc || doc.kind !== "lesson") {
    elements.completeButton.hidden = true;
    return;
  }
  const complete = state.completed.has(doc.id);
  elements.completeButton.hidden = false;
  elements.completeButton.classList.toggle("is-complete", complete);
  elements.completeButton.innerHTML = complete
    ? '<span aria-hidden="true">●</span><span>已完成</span>'
    : '<span aria-hidden="true">○</span><span>标记完成</span>';
  elements.completeButton.setAttribute("aria-pressed", String(complete));
}

function updateNavigationCompletion() {
  for (const link of elements.navigation.querySelectorAll(".navigation-link")) {
    const doc = state.documentByPath.get(normalizePath(link.dataset.path));
    const complete = Boolean(doc?.kind === "lesson" && state.completed.has(doc.id));
    link.classList.toggle("is-complete", complete);
    const check = link.querySelector(".navigation-check");
    if (check && doc?.kind === "lesson") check.textContent = complete ? "●" : "○";
  }
}

function updateProgressSummary() {
  const lessons = state.allDocuments.filter((doc) => doc.kind === "lesson");
  const completed = lessons.filter((doc) => state.completed.has(doc.id)).length;
  const percent = lessons.length ? (completed / lessons.length) * 100 : 0;
  elements.progressText.textContent = `${completed} / ${lessons.length}`;
  elements.progressBar.style.width = `${percent}%`;
}

function loadProgress() {
  try {
    const key = state.manifest.course.storage_key;
    const value = JSON.parse(localStorage.getItem(key) || "[]");
    const validIds = new Set(
      state.allDocuments.filter((doc) => doc.kind === "lesson").map((doc) => doc.id),
    );
    return new Set(Array.isArray(value) ? value.filter((id) => validIds.has(id)) : []);
  } catch {
    return new Set();
  }
}

function persistProgress() {
  localStorage.setItem(
    state.manifest.course.storage_key,
    JSON.stringify([...state.completed].sort()),
  );
}

function navigateRelative(offset) {
  const index = state.allDocuments.findIndex(
    (doc) => normalizePath(doc.path) === state.currentPath,
  );
  const target = state.allDocuments[index + offset];
  if (target) loadDocument(target.path);
}

function setSidebarOpen(open) {
  elements.sidebar.classList.toggle("is-open", open);
  elements.menuButton.setAttribute("aria-expanded", String(open));
  elements.backdrop.hidden = !open;
}

function setSearchOpen(open) {
  elements.search.parentElement.classList.toggle("is-open", open);
  elements.searchButton.setAttribute("aria-expanded", String(open));
  if (open) requestAnimationFrame(() => elements.search.focus());
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
