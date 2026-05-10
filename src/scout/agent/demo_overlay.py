"""Demo overlay — a separate browser window that visualizes the agent's actions.

When ``demo=True``, Scout opens a popup window beside the website and
renders agent actions (thinking, tool calls, results) in real time.

Architecture:
  - The overlay runs in a **separate browser window** (popup), not
    inside the website's DOM.  This eliminates overlap, navigation
    wipes, click interference, and sanitizer stripping.
  - ``BrowserManager`` creates and positions the popup window.
  - ``DemoOverlay`` loads the UI via ``set_content()`` and pushes
    events via ``page.evaluate()``.
  - The overlay page never navigates, so ``window.__scout`` is always
    available — no replay-on-navigation logic needed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Overlay HTML — loaded into the popup window via set_content()
# ═══════════════════════════════════════════════════════════════════════════

_OVERLAY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scout</title>
<style>
  /* ── Reset ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    width: 100%; height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 13px;
    line-height: 1.55;
    color: #e5e5e7;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
    background: #161618;
  }

  .root {
    display: flex; flex-direction: column;
    height: 100%; background: #161618;
  }

  /* ── Suppress animations during replay ── */
  .root.no-animate, .root.no-animate *, .root.no-animate *::after {
    animation: none !important; transition: none !important;
    scroll-behavior: auto !important;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px;
    background: rgba(22,22,24,0.95);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .logo {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, #5e5ce6, #bf5af2);
    display: grid; place-items: center;
    font-weight: 600; font-size: 12px; color: #fff; flex-shrink: 0;
  }
  .header-title {
    font-size: 14px; font-weight: 600;
    color: rgba(255,255,255,0.9);
    letter-spacing: -0.01em;
  }
  .header-right { display: flex; align-items: center; gap: 6px; }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #0a84ff;
    animation: pulse 2s ease-in-out infinite;
  }
  .status-text {
    font-size: 11px; color: rgba(255,255,255,0.35); font-weight: 500;
  }

  /* ── Animations ── */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  @keyframes spin  { to{transform:rotate(360deg)} }

  /* ── Scroll area ── */
  .feed {
    flex: 1; overflow-y: auto; overscroll-behavior: contain;
    padding: 6px 0; scroll-behavior: smooth;
  }
  .feed::-webkit-scrollbar { width: 4px; }
  .feed::-webkit-scrollbar-track { background: transparent; }
  .feed::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 4px; }
  .feed::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

  /* ═══════════ Thinking (main content) ═══════════ */
  .think { padding: 6px 16px; }
  .think-body {
    font-size: 13px; line-height: 1.6;
    color: #d1d1d6; font-weight: 400;
    max-height: 200px; overflow: hidden;
    position: relative;
    transition: max-height 0.4s cubic-bezier(0.4,0,0.2,1);
  }
  .think-body.clipped::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 48px;
    background: linear-gradient(transparent, #161618);
    pointer-events: none;
  }
  .think.open .think-body { max-height: 10000px; }
  .think.open .think-body.clipped::after { display: none; }

  /* Markdown rendered content */
  .think-body strong { color: #f5f5f7; font-weight: 600; }
  .think-body em     { font-style: italic; color: #aeaeb2; }
  .think-body del    { text-decoration: line-through; color: #8e8e93; }
  .think-body a {
    color: #64d2ff; text-decoration: none;
    border-bottom: 1px solid rgba(100,210,255,0.25);
  }
  .think-body a:hover { border-bottom-color: rgba(100,210,255,0.6); }

  .think-body code {
    background: rgba(255,255,255,0.07); padding: 1px 5px;
    border-radius: 4px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 11.5px; color: #a1a1a6;
  }
  .think-body pre {
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px 12px; margin: 8px 0;
    overflow-x: auto; white-space: pre;
  }
  .think-body pre::-webkit-scrollbar { height: 3px; }
  .think-body pre::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
  .think-body pre code {
    background: none; padding: 0; border-radius: 0;
    color: #98989d; font-size: 11px; white-space: pre;
  }

  .think-body h1, .think-body h2, .think-body h3 {
    color: #f5f5f7; font-weight: 600; margin: 14px 0 4px; line-height: 1.3;
  }
  .think-body h1 { font-size: 15px; }
  .think-body h2 { font-size: 14px; }
  .think-body h3 { font-size: 13px; color: #e5e5e7; }

  .think-body p { margin: 0 0 6px; }
  .think-body p:last-child { margin-bottom: 0; }

  .think-body ul, .think-body ol {
    margin: 4px 0 8px; padding-left: 20px; list-style: none;
  }
  .think-body ol { counter-reset: ol-counter; }
  .think-body ul > li {
    position: relative; margin-bottom: 2px;
  }
  .think-body ul > li::before {
    content: '\2022'; position: absolute; left: -15px;
    color: rgba(255,255,255,0.22);
  }
  .think-body ol > li {
    counter-increment: ol-counter; position: relative;
    margin-bottom: 2px;
  }
  .think-body ol > li::before {
    content: counter(ol-counter) '.';
    position: absolute; left: -22px;
    color: rgba(255,255,255,0.3);
    font-size: 12px; font-variant-numeric: tabular-nums;
  }
  .think-body li.cb { display: flex; align-items: flex-start; gap: 6px; }
  .think-body li.cb::before { display: none; }
  .cb-box {
    width: 13px; height: 13px; border-radius: 3px;
    border: 1.5px solid rgba(255,255,255,0.18);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 8px; color: transparent; flex-shrink: 0; margin-top: 3px;
  }
  .cb-box.checked {
    background: rgba(48,209,88,0.15); border-color: rgba(48,209,88,0.4);
    color: #30d158;
  }
  .think-body blockquote {
    border-left: 2px solid rgba(255,255,255,0.08);
    padding-left: 12px; color: #8e8e93; margin: 6px 0;
    font-style: italic;
  }
  .think-body hr {
    border: none; height: 1px;
    background: rgba(255,255,255,0.06); margin: 12px 0;
  }

  /* ═══════════ Action card (tool calls) ═══════════ */
  .action { padding: 3px 16px; }
  .action-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px; cursor: pointer;
    transition: background 0.15s ease;
    user-select: none; -webkit-user-select: none;
  }
  .action-row:hover { background: rgba(255,255,255,0.05); }
  .action-row:active { background: rgba(255,255,255,0.02); }

  .action-ind {
    width: 16px; height: 16px; flex-shrink: 0;
    display: grid; place-items: center;
    font-size: 0; line-height: 1;
  }
  .action-ind.loading::after {
    content: ''; width: 12px; height: 12px;
    border: 1.5px solid rgba(255,255,255,0.06);
    border-top-color: rgba(255,255,255,0.4);
    border-radius: 50%; animation: spin 0.7s linear infinite;
  }
  .action-ind.ok  { color: #30d158; font-size: 13px; }
  .action-ind.err { color: #ff453a; font-size: 13px; }

  .action-label {
    flex: 1; font-size: 12.5px;
    color: #98989d; font-weight: 500;
  }
  .action-meta {
    font-size: 11px; color: rgba(255,255,255,0.18);
    font-variant-numeric: tabular-nums;
  }
  .action-chevron {
    font-size: 14px; color: rgba(255,255,255,0.12);
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1), color 0.15s;
    font-weight: 300; line-height: 1;
  }
  .action-row:hover .action-chevron { color: rgba(255,255,255,0.25); }
  .action.open .action-chevron { transform: rotate(90deg); }

  .action-detail {
    overflow: hidden; max-height: 0; opacity: 0;
    transition: max-height 0.4s cubic-bezier(0.4,0,0.2,1),
                opacity 0.25s ease, padding 0.3s ease;
    padding: 0;
  }
  .action.open .action-detail {
    max-height: 10000px; opacity: 1; padding: 8px 0 2px;
  }
  .action-code {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 8px; padding: 10px 12px; margin: 0;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 11px; line-height: 1.5;
    color: rgba(255,255,255,0.5);
    white-space: pre; overflow-x: auto;
  }
  .action-code::-webkit-scrollbar { height: 3px; }
  .action-code::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
  .action-out {
    margin-top: 6px; padding: 8px 12px;
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 8px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 11px; line-height: 1.45; color: #8e8e93;
    white-space: pre; overflow-x: auto;
    max-height: 300px; overflow-y: auto;
  }
  .action-out::-webkit-scrollbar { width: 3px; height: 3px; }
  .action-out::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
  .action-out.e { color: rgba(255,69,58,0.75); }

  /* Syntax highlighting */
  .action-code .kw  { color: #ff6482; }
  .action-code .bi  { color: #78b6ff; }
  .action-code .fn  { color: #bf9eff; }
  .action-code .str { color: #6cd68e; }
  .action-code .num { color: #78b6ff; }
  .action-code .cm  { color: rgba(255,255,255,0.2); font-style: italic; }
  .action-code .op  { color: rgba(255,255,255,0.4); }
  .action-code .dec { color: #ffa657; }

  /* ═══════════ Turn separator ═══════════ */
  .turn {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px;
    color: rgba(255,255,255,0.12); font-size: 10px;
    font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase;
  }
  .turn-line { flex: 1; height: 1px; background: rgba(255,255,255,0.04); }

  /* ═══════════ System message ═══════════ */
  .sys {
    padding: 3px 16px;
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: #636366;
  }
  .sys-dot {
    width: 3px; height: 3px; border-radius: 50%;
    background: #48484a; flex-shrink: 0;
  }

  /* ═══════════ Terminal state ═══════════ */
  .terminal { padding: 12px 16px; }
  .terminal-card {
    padding: 24px 20px; border-radius: 14px;
  }
  .terminal-card.success {
    background: rgba(48,209,88,0.04);
    border: 1px solid rgba(48,209,88,0.1);
  }
  .terminal-card.fail {
    background: rgba(255,69,58,0.04);
    border: 1px solid rgba(255,69,58,0.1);
  }
  .terminal-icon {
    width: 28px; height: 28px; border-radius: 50%;
    display: inline-grid; place-items: center;
    font-size: 14px; font-weight: 600; margin-bottom: 8px;
  }
  .terminal-card.success .terminal-icon {
    background: rgba(48,209,88,0.08); color: #30d158;
  }
  .terminal-card.fail .terminal-icon {
    background: rgba(255,69,58,0.08); color: #ff453a;
  }
  .terminal-title {
    font-size: 14px; font-weight: 600; color: #e5e5e7;
  }
  .terminal-sub {
    font-size: 12px; color: #8e8e93;
    margin-top: 8px; line-height: 1.5; text-align: left;
  }
  .terminal-sub ul, .terminal-sub ol { margin: 4px 0; padding-left: 18px; }
  .terminal-sub li { margin: 2px 0; }
  .terminal-sub code {
    background: rgba(255,255,255,0.06); padding: 1px 5px;
    border-radius: 3px; font-size: 11px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .terminal-sub pre {
    background: rgba(0,0,0,0.3); padding: 8px 10px;
    border-radius: 6px; overflow-x: auto; margin: 6px 0;
    font-size: 11px; line-height: 1.45;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }

  /* ═══════════ Plan card ═══════════ */
  .plan { padding: 4px 16px; }
  .plan-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; overflow: hidden;
  }
  .plan-header {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px; cursor: pointer;
    user-select: none; -webkit-user-select: none;
    transition: background 0.15s;
  }
  .plan-header:hover { background: rgba(255,255,255,0.025); }
  .plan-icon { font-size: 13px; flex-shrink: 0; }
  .plan-title {
    flex: 1; font-size: 11px; font-weight: 600;
    color: #8e8e93; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .plan-chevron {
    font-size: 14px; color: rgba(255,255,255,0.12);
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
    font-weight: 300; line-height: 1;
  }
  .plan.open .plan-chevron { transform: rotate(90deg); }
  .plan-body {
    max-height: 0; overflow: hidden;
    transition: max-height 0.35s cubic-bezier(0.4,0,0.2,1), opacity 0.25s;
    opacity: 0;
  }
  .plan.open .plan-body { max-height: 600px; opacity: 1; overflow-y: auto; }
  .plan-body .plan-content {
    padding: 4px 12px 10px; font-size: 12px; color: #8e8e93; line-height: 1.5;
  }
  .plan-body .plan-content ul, .plan-body .plan-content ol { margin: 4px 0; padding-left: 18px; list-style: none; }
  .plan-body .plan-content ul > li { position: relative; margin-bottom: 3px; }
  .plan-body .plan-content ul > li::before {
    content: '\2022'; position: absolute; left: -15px;
    color: rgba(255,255,255,0.22);
  }
  .plan-body .plan-content li.cb { display: flex; align-items: flex-start; gap: 6px; }
  .plan-body .plan-content li.cb::before { display: none; }
  .plan-body .plan-content ul ul { margin: 2px 0 4px 12px; }
  .plan-body .plan-content code {
    background: rgba(255,255,255,0.06); padding: 1px 5px;
    border-radius: 3px; font-size: 11px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .plan-body .plan-content pre {
    background: rgba(0,0,0,0.3); padding: 8px 10px;
    border-radius: 6px; overflow-x: auto; margin: 6px 0;
    font-size: 11px; line-height: 1.45;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .plan-body .plan-content hr {
    border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 8px 0;
  }
  .plan-progress {
    padding: 10px 12px;
  }
  .plan-progress-label {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 6px;
  }
  .plan-progress-text {
    font-size: 11.5px; color: #8e8e93; font-weight: 500;
  }
  .plan-progress-pct {
    font-size: 10px; color: rgba(255,255,255,0.2);
    font-variant-numeric: tabular-nums;
  }
  .plan-progress-track {
    width: 100%; height: 3px; border-radius: 2px;
    background: rgba(255,255,255,0.06); overflow: hidden;
  }
  .plan-progress-fill {
    height: 100%; border-radius: 2px; width: 0%;
    background: linear-gradient(90deg, #5e5ce6, #bf5af2);
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .plan-progress-fill.complete {
    background: linear-gradient(90deg, #30d158, #34c759);
    transition: width 0.3s cubic-bezier(0.0, 0, 0.2, 1);
  }

  /* ═══════════ Validation card ═══════════ */
  .validation { padding: 3px 16px; }
  .validation-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
  }
  .validation-ind {
    width: 16px; height: 16px; flex-shrink: 0;
    display: grid; place-items: center; font-size: 0; line-height: 1;
  }
  .validation-ind.loading::after {
    content: ''; width: 12px; height: 12px;
    border: 1.5px solid rgba(255,255,255,0.06);
    border-top-color: rgba(255,255,255,0.4);
    border-radius: 50%; animation: spin 0.7s linear infinite;
  }
  .validation-ind.ok  { color: #30d158; font-size: 13px; }
  .validation-ind.err { color: #ff453a; font-size: 13px; }
  .validation-label { flex: 1; font-size: 12.5px; color: #98989d; font-weight: 500; }
  .validation-detail {
    margin-top: 4px; padding: 6px 12px;
    font-size: 11.5px; color: #8e8e93; line-height: 1.5; text-align: left;
  }
  .validation-detail ul, .validation-detail ol { margin: 4px 0; padding-left: 18px; }
  .validation-detail li { margin: 2px 0; }
  .validation-detail code {
    background: rgba(255,255,255,0.06); padding: 1px 5px;
    border-radius: 3px; font-size: 10.5px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .validation-detail pre {
    background: rgba(0,0,0,0.3); padding: 8px 10px;
    border-radius: 6px; overflow-x: auto; margin: 6px 0;
    font-size: 10.5px; line-height: 1.45;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }

  /* ═══════════ Boot sequence ═══════════ */
  .boot { padding: 14px 16px 4px; }
  .boot-line {
    display: flex; align-items: center; gap: 8px;
    padding: 3px 0; font-size: 11.5px; color: #636366;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    opacity: 0; animation: bootIn 0.35s ease forwards;
  }
  .boot-line .boot-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: #5e5ce6; flex-shrink: 0;
  }
  .boot-line.done .boot-dot { background: #30d158; }
  .boot-active {
    display: flex; align-items: center; gap: 8px;
    padding: 3px 0; font-size: 11.5px; color: #8e8e93;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .boot-active::before {
    content: ''; width: 10px; height: 10px;
    border: 1.5px solid rgba(94,92,230,0.15);
    border-top-color: #5e5ce6;
    border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
  }
  @keyframes bootIn {
    from { opacity:0; transform:translateY(3px); }
    to   { opacity:1; transform:translateY(0); }
  }
</style>
</head>
<body>
<div class="root" id="root">
  <div class="header">
    <div class="header-left">
      <div class="logo">S</div>
      <span class="header-title">Scout</span>
    </div>
    <div class="header-right">
      <div class="status-dot" id="dot"></div>
      <span class="status-text" id="status-text">Running</span>
    </div>
  </div>
  <div class="feed" id="feed"></div>
</div>
</body>
</html>"""

# JavaScript controller — injected via page.evaluate() after set_content()
# because Playwright's set_content() does not execute inline <script> tags.
_OVERLAY_JS = r"""
(() => {
  const root = document.getElementById('root');
  const feed = document.getElementById('feed');

  /* ── Helpers ─────────────────────────────────────────────────── */
  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  let _replaying = false;
  let _userNearBottom = true;
  feed.addEventListener('scroll', () => {
    const threshold = 80;
    _userNearBottom = (feed.scrollHeight - feed.scrollTop - feed.clientHeight) < threshold;
  }, { passive: true });
  function scrollDown() {
    if (_replaying) return;
    if (!_userNearBottom) return;
    requestAnimationFrame(() => { feed.scrollTop = feed.scrollHeight; });
  }
  function trim() {
    while (feed.children.length > 120) feed.removeChild(feed.firstChild);
  }

  /* ═══════════ Markdown parser ═══════════
   * Token placeholders use `%%TOKn%%` (safe through DOM operations).
   * Line-based parsing handles: headers, hr, lists, checkboxes,
   * blockquotes, fenced code, inline code, bold, italic,
   * strikethrough, links.
   */
  function md(text) {
    const tokens = [];
    let src = text;
    const PH = '%%TOK';

    // Extract fenced code blocks
    src = src.replace(/```(\w*)\n([\s\S]*?)```/g, function(_, lang, code) {
      const i = tokens.length;
      tokens.push('<pre><code>' + esc(code.replace(/\n$/, '')) + '</code></pre>');
      return PH + i + '%%';
    });

    // Extract inline code
    src = src.replace(/`([^`]+)`/g, function(_, code) {
      const i = tokens.length;
      tokens.push('<code>' + esc(code) + '</code>');
      return PH + i + '%%';
    });

    // Inline formatting on already-escaped text
    function inl(h) {
      // Restore token placeholders first
      for (let i = 0; i < tokens.length; i++) {
        h = h.split(PH + i + '%%').join(tokens[i]);
      }
      // Bold+italic
      h = h.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
      h = h.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');
      // Bold
      h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/__(.+?)__/g, '<strong>$1</strong>');
      // Italic (careful not to match inside words for underscores)
      h = h.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
      h = h.replace(/(?<![a-zA-Z0-9])_(?!_)(.+?)(?<!_)_(?![a-zA-Z0-9])/g, '<em>$1</em>');
      // Strikethrough
      h = h.replace(/~~(.+?)~~/g, '<del>$1</del>');
      // Links [text](url)
      h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
      return h;
    }

    // Process line-by-line
    const lines = src.split('\n');
    const out = [];
    let inUl = false, inOl = false;

    function closeList() {
      if (inUl) { out.push('</ul>'); inUl = false; }
      if (inOl) { out.push('</ol>'); inOl = false; }
    }

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i];
      const trimmed = raw.trim();

      // Token placeholder line (standalone code block)
      const tokMatch = trimmed.match(/^%%TOK(\d+)%%$/);
      if (tokMatch) { closeList(); out.push(tokens[parseInt(tokMatch[1])]); continue; }

      // Horizontal rule: ---, ***, ___  (3+ chars, optionally spaced)
      if (/^[-]{3,}$/.test(trimmed) || /^[*]{3,}$/.test(trimmed) || /^[_]{3,}$/.test(trimmed)) {
        closeList(); out.push('<hr>'); continue;
      }

      // Headers
      const hm = raw.match(/^(#{1,3}) (.+)$/);
      if (hm) { closeList(); out.push('<h' + hm[1].length + '>' + inl(esc(hm[2])) + '</h' + hm[1].length + '>'); continue; }

      // Blockquote
      const bq = raw.match(/^> ?(.*)$/);
      if (bq) { closeList(); out.push('<blockquote>' + inl(esc(bq[1])) + '</blockquote>'); continue; }

      // Checkbox list item
      const cb = raw.match(/^[ \t]*[-*+] \[([ xX])\] (.+)$/);
      if (cb) {
        if (!inUl) { closeList(); out.push('<ul>'); inUl = true; }
        const chk = cb[1] !== ' ';
        out.push('<li class="cb"><span class="cb-box' + (chk ? ' checked' : '') + '">'
          + (chk ? '\u2713' : '') + '</span>' + inl(esc(cb[2])) + '</li>');
        continue;
      }

      // Unordered list item
      const ul = raw.match(/^[ \t]*[-*+] (.+)$/);
      if (ul) {
        if (!inUl) { closeList(); out.push('<ul>'); inUl = true; }
        out.push('<li>' + inl(esc(ul[1])) + '</li>');
        continue;
      }

      // Ordered list item
      const ol = raw.match(/^[ \t]*\d+[.)]\s(.+)$/);
      if (ol) {
        if (!inOl) { closeList(); out.push('<ol>'); inOl = true; }
        out.push('<li>' + inl(esc(ol[1])) + '</li>');
        continue;
      }

      closeList();

      // Empty line
      if (trimmed === '') continue;

      // Regular paragraph
      out.push('<p>' + inl(esc(raw)) + '</p>');
    }
    closeList();
    return out.join('');
  }

  /* ═══════════ Python syntax highlighter ═══════════ */
  const KW = new Set([
    'False','None','True','and','as','assert','async','await',
    'break','class','continue','def','del','elif','else','except',
    'finally','for','from','global','if','import','in','is',
    'lambda','nonlocal','not','or','pass','raise','return',
    'try','while','with','yield',
  ]);
  const BI = new Set([
    'print','len','range','int','str','float','list','dict','set',
    'tuple','bool','type','isinstance','enumerate','zip','map',
    'filter','sorted','reversed','any','all','min','max','sum',
    'abs','round','open','super','property','staticmethod',
    'classmethod','hasattr','getattr','setattr','delattr',
    'next','iter','input','format','repr','hash','id','vars',
    'Exception','ValueError','TypeError','KeyError','IndexError',
    'AttributeError','RuntimeError','StopIteration','IOError',
    'FileNotFoundError','NotImplementedError','TimeoutError',
  ]);

  function hiPy(code) {
    const tq = '"{3}';
    const re = new RegExp(
      '(' + tq + '[\\s\\S]*?' + tq +
      "|'{3}[\\s\\S]*?'{3}" +
      '|"(?:[^"\\\\]|\\\\.)*"' +
      "|'(?:[^'\\\\]|\\\\.)*'" +
      '|f"(?:[^"\\\\]|\\\\.)*"' +
      "|f'(?:[^'\\\\]|\\\\.)*'" +
      ')|(\\#[^\\n]*)' +
      '|(\\b\\d+(?:\\.\\d+)?(?:e[+-]?\\d+)?\\b)' +
      '|(@\\w+)' +
      '|(\\b\\w+(?=\\s*\\())|\\b(\\w+)\\b' +
      '|([+\\-*/%=<>!&|^~:]+|[()\\[\\]{},;.])',
      'g'
    );
    const out = [];
    let last = 0, m;
    while ((m = re.exec(code)) !== null) {
      if (m.index > last) out.push(esc(code.slice(last, m.index)));
      last = m.index + m[0].length;
      const raw = esc(m[0]);
      if      (m[1]) out.push('<span class="str">' + raw + '</span>');
      else if (m[2]) out.push('<span class="cm">'  + raw + '</span>');
      else if (m[3]) out.push('<span class="num">' + raw + '</span>');
      else if (m[4]) out.push('<span class="dec">' + raw + '</span>');
      else if (m[5]) {
        out.push(KW.has(m[5]) ? '<span class="kw">' + raw + '</span>' :
                 BI.has(m[5]) ? '<span class="bi">' + raw + '</span>' :
                 '<span class="fn">' + raw + '</span>');
      }
      else if (m[6]) {
        out.push(KW.has(m[6]) ? '<span class="kw">' + raw + '</span>' :
                 BI.has(m[6]) ? '<span class="bi">' + raw + '</span>' : raw);
      }
      else if (m[7]) out.push('<span class="op">' + raw + '</span>');
      else out.push(raw);
    }
    if (last < code.length) out.push(esc(code.slice(last)));
    return out.join('');
  }

  function inferLabel(code) {
    if (!code) return 'Interacting with the website';
    const c = code.toLowerCase();
    if (c.includes('show_page') || c.includes('get_page_html') || c.includes('page_html'))
      return 'Looking at the website';
    if (c.includes('show_section') || c.includes('get_section'))
      return 'Inspecting the page';
    return 'Interacting with the website';
  }

  /* ═══════════ State ═══════════ */
  let lastAction = null;

  function setStatus(text, color) {
    const st = document.getElementById('status-text');
    const dot = document.getElementById('dot');
    if (st) st.textContent = text;
    if (dot) { dot.style.animation = 'none'; dot.style.background = color; }
  }

  /* ═══════════ Controller ═══════════ */
  window.__scout = {
    replayEvents(events) {
      feed.innerHTML = '';
      root.classList.add('no-animate');
      lastAction = null;
      _replaying = true;
      for (const ev of events) this.pushEvent(ev);
      _replaying = false;
      feed.scrollTop = feed.scrollHeight;
      void root.offsetHeight;
      root.classList.remove('no-animate');
    },

    pushEvent(ev) {
      const t = ev.type;

      if (t === 'turn') {
        const d = document.createElement('div');
        d.className = 'turn';
        d.innerHTML = '<span class="turn-line"></span>Turn ' + ev.turn
                    + '<span class="turn-line"></span>';
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'thinking') {
        const entry = document.createElement('div');
        entry.className = 'think';
        const body = document.createElement('div');
        body.className = 'think-body';
        body.innerHTML = md(ev.text || '');
        entry.appendChild(body);
        feed.appendChild(entry); trim(); scrollDown();
        requestAnimationFrame(() => {
          if (body.scrollHeight > body.clientHeight + 4) {
            body.classList.add('clipped');
            entry.style.cursor = 'pointer';
            entry.addEventListener('click', () => entry.classList.toggle('open'));
          }
        });
        return;
      }

      if (t === 'tool_call') {
        const label = ev.label || inferLabel(ev.code);
        const entry = document.createElement('div');
        entry.className = 'action';
        const row = document.createElement('div');
        row.className = 'action-row';
        const ind = document.createElement('div');
        ind.className = 'action-ind loading';
        const lbl = document.createElement('span');
        lbl.className = 'action-label';
        lbl.textContent = label;
        const meta = document.createElement('span');
        meta.className = 'action-meta';
        if (ev.step && ev.max_steps) meta.textContent = ev.step + '/' + ev.max_steps;
        const chev = document.createElement('span');
        chev.className = 'action-chevron';
        chev.textContent = '\u203A';
        row.append(ind, lbl, meta, chev);
        const detail = document.createElement('div');
        detail.className = 'action-detail';
        if (ev.code) {
          const pre = document.createElement('pre');
          pre.className = 'action-code';
          pre.innerHTML = hiPy(ev.code);
          detail.appendChild(pre);
        }
        entry.append(row, detail);
        row.addEventListener('click', () => entry.classList.toggle('open'));
        feed.appendChild(entry); lastAction = entry; trim(); scrollDown();
        return;
      }

      if (t === 'tool_result') {
        const target = lastAction;
        if (target) {
          const ind = target.querySelector('.action-ind');
          ind.className = 'action-ind ' + (ev.is_error ? 'err' : 'ok');
          ind.textContent = ev.is_error ? '\u2717' : '\u2713';
          const meta = target.querySelector('.action-meta');
          if (ev.duration_s) meta.textContent = ev.duration_s + 's';
          const detail = target.querySelector('.action-detail');
          if (ev.output) {
            const o = document.createElement('div');
            o.className = 'action-out';
            o.textContent = ev.output;
            detail.appendChild(o);
          }
          if (ev.error) {
            const e = document.createElement('div');
            e.className = 'action-out e';
            e.textContent = ev.error;
            detail.appendChild(e);
          }
          lastAction = null;
        }
        scrollDown(); return;
      }

      if (t === 'page_update') {
        const d = document.createElement('div');
        d.className = 'sys';
        d.innerHTML = '<span class="sys-dot"></span>Page captured \u2014 '
                    + esc(String(ev.sections || '?')) + ' sections';
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'script_found') {
        const d = document.createElement('div');
        d.className = 'sys';
        d.innerHTML = '<span class="sys-dot"></span>'
          + (ev.valid ? 'Script extracted' : 'Script issue \u2014 ' + esc(ev.error || ''));
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'script_running') {
        const entry = document.createElement('div');
        entry.className = 'action';
        const row = document.createElement('div');
        row.className = 'action-row';
        const ind = document.createElement('div');
        ind.className = 'action-ind loading';
        const lbl = document.createElement('span');
        lbl.className = 'action-label';
        lbl.textContent = 'Running extraction';
        const meta = document.createElement('span');
        meta.className = 'action-meta';
        const chev = document.createElement('span');
        chev.className = 'action-chevron';
        chev.textContent = '\u203A';
        row.append(ind, lbl, meta, chev);
        const detail = document.createElement('div');
        detail.className = 'action-detail';
        entry.append(row, detail);
        row.addEventListener('click', () => entry.classList.toggle('open'));
        feed.appendChild(entry); lastAction = entry; scrollDown();
        return;
      }

      if (t === 'script_output') {
        const target = lastAction;
        if (target) {
          const ok = ev.returncode === 0;
          const ind = target.querySelector('.action-ind');
          ind.className = 'action-ind ' + (ok ? 'ok' : 'err');
          ind.textContent = ok ? '\u2713' : '\u2717';
          if (ev.output) {
            const detail = target.querySelector('.action-detail');
            const o = document.createElement('div');
            o.className = 'action-out' + (ok ? '' : ' e');
            o.textContent = ev.output;
            detail.appendChild(o);
          }
          lastAction = null;
        }
        scrollDown(); return;
      }

      if (t === 'approved') {
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card success">'
          + '<div class="terminal-icon">\u2713</div>'
          + '<div class="terminal-title">Approved</div></div>';
        feed.appendChild(d); setStatus('Complete', '#30d158'); scrollDown();
        return;
      }

      if (t === 'rejected') {
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card fail">'
          + '<div class="terminal-icon">\u2717</div>'
          + '<div class="terminal-title">Needs revision</div>'
          + (ev.feedback ? '<div class="terminal-sub">' + md(ev.feedback) + '</div>' : '')
          + '</div>';
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'done') {
        const ok = ev.success;
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card ' + (ok ? 'success' : 'fail') + '">'
          + '<div class="terminal-icon">' + (ok ? '\u2713' : '\u2717') + '</div>'
          + '<div class="terminal-title">' + (ok ? 'Complete' : 'Failed') + '</div>'
          + (!ok && ev.error ? '<div class="terminal-sub">' + md(ev.error) + '</div>' : '')
          + '</div>';
        feed.appendChild(d);
        setStatus(ok ? 'Complete' : 'Failed', ok ? '#30d158' : '#ff453a');
        scrollDown(); return;
      }

      if (t === 'system') {
        const d = document.createElement('div');
        d.className = 'sys';
        d.innerHTML = '<span class="sys-dot"></span>' + esc(ev.message || '');
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'validation') {
        const d = document.createElement('div');
        d.className = 'validation';
        d.setAttribute('data-vid', ev.id || '');
        const row = document.createElement('div');
        row.className = 'validation-row';
        const ind = document.createElement('div');
        const status = ev.status || 'loading';
        ind.className = 'validation-ind ' + status;
        if (status === 'ok')  ind.textContent = '\u2713';
        if (status === 'err') ind.textContent = '\u2717';
        const lbl = document.createElement('span');
        lbl.className = 'validation-label';
        lbl.textContent = ev.label || '';
        row.append(ind, lbl);
        d.appendChild(row);
        if (ev.detail) {
          const det = document.createElement('div');
          det.className = 'validation-detail';
          det.innerHTML = md(ev.detail);
          d.appendChild(det);
        }
        feed.appendChild(d); scrollDown(); return;
      }

      if (t === 'validation_update') {
        const target = feed.querySelector('.validation[data-vid="' + (ev.id || '') + '"]');
        if (target) {
          const ind = target.querySelector('.validation-ind');
          const status = ev.status || 'ok';
          ind.className = 'validation-ind ' + status;
          if (status === 'ok')  ind.textContent = '\u2713';
          if (status === 'err') ind.textContent = '\u2717';
          if (status === 'loading') ind.textContent = '';
          if (ev.label) target.querySelector('.validation-label').textContent = ev.label;
          if (ev.detail) {
            let det = target.querySelector('.validation-detail');
            if (!det) { det = document.createElement('div'); det.className = 'validation-detail'; target.appendChild(det); }
            det.innerHTML = md(ev.detail);
          }
        }
        scrollDown(); return;
      }

      if (t === 'boot') {
        const old = feed.querySelector('.boot');
        if (old) old.remove();
        if (window.__bootInterval) { clearInterval(window.__bootInterval); window.__bootInterval = null; }
        const messages = ev.messages || [];
        const d = document.createElement('div');
        d.className = 'boot';
        feed.appendChild(d);
        let idx = 0;
        function showNext() {
          if (idx >= messages.length) { clearInterval(window.__bootInterval); window.__bootInterval = null; return; }
          const prev = d.querySelector('.boot-active');
          if (prev) {
            const done = document.createElement('div');
            done.className = 'boot-line done';
            done.innerHTML = '<span class="boot-dot"></span>' + esc(prev.textContent);
            d.replaceChild(done, prev);
          }
          const line = document.createElement('div');
          line.className = 'boot-active';
          line.textContent = messages[idx];
          d.appendChild(line);
          idx++; scrollDown();
        }
        showNext();
        window.__bootInterval = setInterval(showNext, 1800);
        return;
      }

      if (t === 'planning') {
        if (window.__bootInterval) { clearInterval(window.__bootInterval); window.__bootInterval = null; }
        const boot = feed.querySelector('.boot');
        if (boot) {
          const active = boot.querySelector('.boot-active');
          if (active) {
            const done = document.createElement('div');
            done.className = 'boot-line done';
            done.innerHTML = '<span class="boot-dot"></span>' + esc(active.textContent);
            boot.replaceChild(done, active);
          }
        }
        const prev = feed.querySelector('.plan-loading');
        if (prev) prev.remove();
        if (window.__planProgressTimer) { clearInterval(window.__planProgressTimer); window.__planProgressTimer = null; }
        const d = document.createElement('div');
        d.className = 'plan-loading plan';
        d.innerHTML = '<div class="plan-card"><div class="plan-progress">'
          + '<div class="plan-progress-label">'
          + '<span class="plan-progress-text">' + esc(ev.message || 'Planning exploration\u2026') + '</span>'
          + '<span class="plan-progress-pct">0%</span>'
          + '</div>'
          + '<div class="plan-progress-track"><div class="plan-progress-fill"></div></div>'
          + '</div></div>';
        feed.appendChild(d); scrollDown();
        // Asymptotic progress: progress = cap * (1 - e^(-k*t))
        // cap=90, k tuned so it feels fast early then slows
        const fill = d.querySelector('.plan-progress-fill');
        const pctLabel = d.querySelector('.plan-progress-pct');
        const startTime = Date.now();
        const cap = 90;
        const k = 0.12; // controls speed — higher = faster start
        window.__planProgressTimer = setInterval(() => {
          const elapsed = (Date.now() - startTime) / 1000;
          const progress = Math.min(cap, cap * (1 - Math.exp(-k * elapsed)));
          const rounded = Math.round(progress);
          fill.style.width = rounded + '%';
          pctLabel.textContent = rounded + '%';
        }, 200);
        return;
      }

      if (t === 'plan') {
        // Completion sprint — snap progress to 100% then swap in the plan card
        if (window.__planProgressTimer) { clearInterval(window.__planProgressTimer); window.__planProgressTimer = null; }
        const loader = feed.querySelector('.plan-loading');
        if (loader) {
          const fill = loader.querySelector('.plan-progress-fill');
          const pctLabel = loader.querySelector('.plan-progress-pct');
          if (fill) { fill.classList.add('complete'); fill.style.width = '100%'; }
          if (pctLabel) pctLabel.textContent = '100%';
        }
        // Brief pause at 100% so user registers completion, then swap
        const showPlan = () => {
          if (loader) loader.remove();
          const items = ev.items || [];
          const d = document.createElement('div');
          d.className = 'plan open';
          // Build list HTML with nesting support
          let listHtml = '<ul>';
          let inSub = false;
          for (const item of items) {
            const isSub = item.startsWith('  ');
            const text = isSub ? item.trim() : item;
            if (isSub && !inSub) { listHtml += '<ul>'; inSub = true; }
            else if (!isSub && inSub) { listHtml += '</ul>'; inSub = false; }
            const cbMatch = text.match(/^\[([ xX])\] (.+)$/);
            if (cbMatch) {
              const chk = cbMatch[1] !== ' ';
              listHtml += '<li class="cb"><span class="cb-box' + (chk ? ' checked' : '') + '">'
                + (chk ? '\u2713' : '') + '</span>' + esc(cbMatch[2]) + '</li>';
            } else {
              listHtml += '<li>' + esc(text) + '</li>';
            }
          }
          if (inSub) listHtml += '</ul>';
          listHtml += '</ul>';
          let html = '<div class="plan-card"><div class="plan-header">'
            + '<span class="plan-icon">\uD83D\uDCCB</span>'
            + '<span class="plan-title">Exploration Plan</span>'
            + '<span class="plan-chevron">\u203A</span>'
            + '</div><div class="plan-body"><div class="plan-content">'
            + listHtml
            + '</div></div></div>';
          d.innerHTML = html;
          d.querySelector('.plan-header').addEventListener('click', () => d.classList.toggle('open'));
          feed.appendChild(d); scrollDown();
        };
        setTimeout(showPlan, loader ? 400 : 0);
        return;
      }
    },
  };
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Python API
# ═══════════════════════════════════════════════════════════════════════════

class DemoOverlay:
    """Manages a separate browser window that shows agent actions.

    The overlay runs in its own page (popup window), completely
    independent of the website the agent is scraping.  Events are
    pushed via ``page.evaluate()`` on the overlay page.

    The overlay page never navigates, so ``window.__scout`` is always
    available — no replay-on-navigation logic is needed.  The event
    buffer is kept as a safety net for recovery if the overlay page
    is accidentally closed or crashes.
    """

    _MAX_BUFFER = 150

    def __init__(self) -> None:
        self._overlay_page = None
        self._injected = False
        self._events: list[dict[str, Any]] = []

    async def init(self, overlay_page) -> None:
        """Load the overlay UI into the popup page."""
        self._overlay_page = overlay_page
        try:
            await overlay_page.set_content(
                _OVERLAY_HTML, wait_until="load",
            )
            # Playwright's set_content() doesn't execute <script> tags,
            # so we inject the JS controller via evaluate().
            await overlay_page.evaluate(_OVERLAY_JS)
            self._injected = True
            logger.info("[overlay] initialized successfully")
        except Exception:
            logger.warning("[overlay] init failed", exc_info=True)
            self._injected = False

    async def push(self, event: dict[str, Any]) -> None:
        """Buffer the event and push it to the overlay."""
        self._events.append(event)
        if len(self._events) > self._MAX_BUFFER:
            self._events = self._events[-self._MAX_BUFFER:]

        if not self._injected or self._overlay_page is None:
            return

        try:
            await self._overlay_page.evaluate(
                "ev => window.__scout.pushEvent(ev)", event,
            )
        except Exception:
            logger.debug(
                "[overlay] push failed (type=%s), recovering",
                event.get("type", "?"),
            )
            await self._recover()

    async def _recover(self) -> None:
        """Re-create overlay content and replay buffered events."""
        if self._overlay_page is None:
            return
        try:
            await self._overlay_page.set_content(
                _OVERLAY_HTML, wait_until="load",
            )
            await self._overlay_page.evaluate(_OVERLAY_JS)
            if self._events:
                await self._overlay_page.evaluate(
                    "events => window.__scout.replayEvents(events)",
                    self._events,
                )
            self._injected = True
        except Exception:
            logger.debug("Overlay recovery failed", exc_info=True)
            self._injected = False

    # ── Typed push helpers ──────────────────────────────────────────

    async def push_turn(self, turn: int) -> None:
        await self.push({"type": "turn", "turn": turn})

    async def push_thinking(self, text: str) -> None:
        await self.push({"type": "thinking", "text": text})

    async def push_tool_call(
        self,
        code: str,
        step: int = 0,
        max_steps: int = 0,
        *,
        label: str = "",
    ) -> None:
        await self.push({
            "type": "tool_call",
            "code": code,
            "step": step,
            "max_steps": max_steps,
            "label": label,
        })

    async def push_tool_result(
        self,
        *,
        is_error: bool = False,
        duration_s: str = "",
        output: str = "",
        error: str = "",
    ) -> None:
        await self.push({
            "type": "tool_result",
            "is_error": is_error,
            "duration_s": duration_s,
            "output": output,
            "error": error,
        })

    async def push_page_update(self, url: str, sections: int) -> None:
        await self.push({
            "type": "page_update",
            "url": url,
            "sections": sections,
        })

    async def push_script_found(
        self, valid: bool, error: str = "",
    ) -> None:
        await self.push({
            "type": "script_found",
            "valid": valid,
            "error": error,
        })

    async def push_script_running(self) -> None:
        await self.push({"type": "script_running"})

    async def push_script_output(
        self, output: str, returncode: int,
    ) -> None:
        await self.push({
            "type": "script_output",
            "output": output,
            "returncode": returncode,
        })

    async def push_approved(self) -> None:
        await self.push({"type": "approved"})

    async def push_rejected(self, feedback: str) -> None:
        await self.push({"type": "rejected", "feedback": feedback})

    async def push_done(
        self, success: bool, error: str = "",
    ) -> None:
        await self.push({
            "type": "done",
            "success": success,
            "error": error,
        })

    async def push_system(self, message: str) -> None:
        await self.push({"type": "system", "message": message})

    async def push_validation(
        self,
        vid: str,
        label: str,
        *,
        status: str = "loading",
        detail: str = "",
    ) -> None:
        """Push a validation step card (loading, ok, or err)."""
        ev: dict[str, Any] = {
            "type": "validation", "id": vid,
            "label": label, "status": status,
        }
        if detail:
            ev["detail"] = detail
        await self.push(ev)

    async def push_validation_update(
        self,
        vid: str,
        *,
        status: str = "ok",
        label: str = "",
        detail: str = "",
    ) -> None:
        """Update an existing validation step in-place."""
        ev: dict[str, Any] = {
            "type": "validation_update", "id": vid, "status": status,
        }
        if label:
            ev["label"] = label
        if detail:
            ev["detail"] = detail
        await self.push(ev)

    async def push_boot(self, url: str = "", has_schema: bool = False) -> None:
        messages = [
            "Launching browser\u2026",
            f"Navigating to {url}\u2026" if url else "Navigating to target\u2026",
            "Waiting for dynamic content\u2026",
            "Capturing page structure\u2026",
        ]
        if has_schema:
            messages.append("Planning exploration\u2026")
        await self.push({"type": "boot", "messages": messages})

    async def push_planning(self, message: str = "Generating plan\u2026") -> None:
        await self.push({"type": "planning", "message": message})

    async def push_plan(self, items: list[str]) -> None:
        await self.push({"type": "plan", "items": items})
