"use strict";

import {
  escapeHtml,
  isExternalHref,
  resolveRelativePath,
  slugify,
} from "./course-utils.js";

let mathBlocks = [];

export function configureRenderers() {
  if (window.marked) {
    window.marked.setOptions({ gfm: true, breaks: false, mangle: false, headerIds: false });
  }
  if (window.mermaid) {
    const dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: dark ? "dark" : "neutral",
      fontFamily:
        'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif',
      flowchart: { useMaxWidth: true, htmlLabels: true, curve: "basis" },
      sequence: { useMaxWidth: true, wrap: true },
    });
  }
}

export async function renderMarkdown({
  markdown,
  currentPath,
  content,
  documentByPath,
  onNavigate,
}) {
  if (!window.marked) throw new Error("Marked 渲染库未加载。请检查网络后刷新页面。");

  const extracted = extractMathBlocks(markdown);
  mathBlocks = extracted.blocks;
  content.innerHTML = window.marked.parse(extracted.markdown);

  assignHeadingIds(content);
  resolveDocumentLinks(content, currentPath, documentByPath, onNavigate);
  resolveImages(content, currentPath);
  wrapTables(content);
  prepareMermaidBlocks(content);
  addCopyButtons(content);
  await renderDiagrams(content);
  renderMath(content);
}

function extractMathBlocks(markdown) {
  const blocks = [];
  const transformed = markdown.replace(/```math\s*\n([\s\S]*?)```/g, (_match, expression) => {
    const index = blocks.push(expression.trim()) - 1;
    return `<div class="math-display" data-math-index="${index}"></div>`;
  });
  return { markdown: transformed, blocks };
}

function assignHeadingIds(content) {
  const seen = new Map();
  for (const heading of content.querySelectorAll("h1, h2, h3, h4")) {
    const base = slugify(heading.textContent || "section");
    const count = seen.get(base) || 0;
    seen.set(base, count + 1);
    heading.id = count === 0 ? base : `${base}-${count + 1}`;
  }
}

function resolveDocumentLinks(content, currentPath, documentByPath, onNavigate) {
  for (const anchor of content.querySelectorAll("a[href]")) {
    const original = anchor.getAttribute("href");
    if (!original) continue;

    if (original.startsWith("#")) {
      const fragment = decodeFragment(original.slice(1));
      anchor.addEventListener("click", (event) => {
        const target = document.getElementById(fragment) || document.getElementById(slugify(fragment));
        if (!target) return;
        event.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        const url = new URL(location.href);
        url.hash = target.id;
        history.replaceState({}, "", url);
      });
      continue;
    }

    if (isExternalHref(original)) {
      anchor.target = "_blank";
      anchor.rel = "noreferrer";
      continue;
    }

    const [pathPart, rawFragment = ""] = original.split("#", 2);
    const fragment = decodeFragment(rawFragment);
    const resolved = resolveRelativePath(currentPath, pathPart);

    if (resolved.endsWith(".md") && documentByPath.has(resolved)) {
      anchor.href = `?doc=${encodeURIComponent(resolved)}${
        fragment ? `#${encodeURIComponent(fragment)}` : ""
      }`;
      anchor.addEventListener("click", (event) => {
        event.preventDefault();
        onNavigate(resolved, fragment);
      });
    } else {
      anchor.href = resolved + (fragment ? `#${fragment}` : "");
    }
  }
}

function resolveImages(content, currentPath) {
  for (const image of content.querySelectorAll("img[src]")) {
    const source = image.getAttribute("src");
    if (!source || isExternalHref(source) || source.startsWith("data:")) continue;
    image.src = resolveRelativePath(currentPath, source);
    image.loading = "lazy";
  }
}

function wrapTables(content) {
  for (const table of [...content.querySelectorAll("table")]) {
    if (table.parentElement?.classList.contains("table-scroll")) continue;
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    table.replaceWith(wrapper);
    wrapper.append(table);
  }
}

function prepareMermaidBlocks(content) {
  for (const code of content.querySelectorAll("pre code.language-mermaid")) {
    const diagram = document.createElement("div");
    diagram.className = "mermaid";
    diagram.textContent = code.textContent || "";
    code.parentElement.replaceWith(diagram);
  }
}

async function renderDiagrams(content) {
  const diagrams = [...content.querySelectorAll(".mermaid")];
  if (!diagrams.length) return;
  if (!window.mermaid) {
    diagrams.forEach((node) => {
      node.className = "render-error";
      node.textContent = `Mermaid 渲染库未加载。\n\n${node.textContent}`;
    });
    return;
  }

  try {
    await window.mermaid.run({ nodes: diagrams });
  } catch (error) {
    console.error("Mermaid render failed", error);
    for (const node of diagrams.filter((item) => !item.querySelector("svg"))) {
      node.className = "render-error";
      node.textContent = `Mermaid 图渲染失败：${errorMessage(error)}`;
    }
  }
}

function renderMath(content) {
  for (const node of content.querySelectorAll("[data-math-index]")) {
    const expression = mathBlocks[Number(node.dataset.mathIndex)] || "";
    if (!window.katex) {
      node.className = "render-error";
      node.textContent = expression;
      continue;
    }
    try {
      window.katex.render(expression, node, {
        displayMode: true,
        throwOnError: false,
        strict: "ignore",
      });
    } catch (error) {
      node.className = "render-error";
      node.textContent = `公式渲染失败：${expression}\n${errorMessage(error)}`;
    }
  }

  window.renderMathInElement?.(content, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false,
  });
}

function addCopyButtons(content) {
  for (const pre of content.querySelectorAll("pre")) {
    const code = pre.querySelector("code");
    if (!code) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-button";
    button.textContent = "复制";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || "");
        button.textContent = "已复制";
        setTimeout(() => (button.textContent = "复制"), 1200);
      } catch {
        button.textContent = "复制失败";
      }
    });
    pre.append(button);
  }
}

export function buildOutline(content, outline) {
  outline.replaceChildren();
  for (const heading of content.querySelectorAll("h2, h3")) {
    const link = document.createElement("a");
    link.className = "outline-link";
    link.dataset.level = heading.tagName.slice(1);
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent || "";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      heading.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState({}, "", `${location.search}#${heading.id}`);
    });
    outline.append(link);
  }
}

export function showDocumentError(content, message) {
  content.innerHTML = `<pre class="render-error">${escapeHtml(message)}</pre>`;
}

export function showFatalError(content, error) {
  content.innerHTML = `
    <h1>课程入口未能初始化</h1>
    <pre class="render-error">${escapeHtml(errorMessage(error))}</pre>
    <p>请从仓库根目录运行：</p>
    <pre><code>python -m http.server --directory course/aios-ime 8000</code></pre>
    <p>然后打开 <code>http://localhost:8000/</code>，不要直接双击 <code>index.html</code>。</p>
  `;
}

function decodeFragment(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
