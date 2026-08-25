"use strict";

export function normalizePath(value) {
  const parts = [];
  for (const part of String(value || "").replaceAll("\\", "/").split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") parts.pop();
    else parts.push(part);
  }
  return parts.join("/");
}

export function resolveRelativePath(currentPath, href) {
  const base = normalizePath(currentPath).split("/");
  base.pop();
  for (const part of String(href || "").replaceAll("\\", "/").split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") base.pop();
    else base.push(part);
  }
  return normalizePath(base.join("/"));
}

export function isExternalHref(href) {
  return /^(?:[a-z]+:)?\/\//i.test(href) || /^(mailto|tel):/i.test(href);
}

export function isEditableTarget(target) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target?.isContentEditable
  );
}

export function slugify(value) {
  const result = String(value || "")
    .trim()
    .toLocaleLowerCase("zh-CN")
    .replace(/[`*_~()[\]{}<>:：，。！？；;“”‘’…]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
  return result || `section-${crypto.randomUUID().slice(0, 8)}`;
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`无法读取 ${path}：HTTP ${response.status}`);
  return response.json();
}

export async function fetchText(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`无法读取 ${path}：HTTP ${response.status}`);
  return response.text();
}

export function requestedDocument() {
  const value = new URLSearchParams(location.search).get("doc");
  return value ? normalizePath(value) : "";
}

export function decodedFragment() {
  try {
    return decodeURIComponent(location.hash.slice(1));
  } catch {
    return location.hash.slice(1);
  }
}

export function findFragmentTarget(fragment) {
  if (!fragment) return null;
  return document.getElementById(fragment) || document.getElementById(slugify(fragment));
}

export function scrollToFragment(fragment, behavior = "smooth") {
  const target = findFragmentTarget(fragment);
  if (target) target.scrollIntoView({ behavior, block: "start" });
}
