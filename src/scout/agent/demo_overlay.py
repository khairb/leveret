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

import asyncio
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .selector_extractor import ExtractionResult

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

  /* ── Agent status indicator (reasoning / generating plan) ── */
  .agent-status {
    padding: 3px 16px 3px 24px;
    display: flex; align-items: center; gap: 0;
    font-size: 11px; color: #636366;
    opacity: 0;
    animation: fade-in 0.3s ease 0.8s forwards;
  }
  @keyframes fade-in { to { opacity: 1; } }
  .agent-status-text span {
    display: inline-block;
    animation: char-wave 2.4s ease-in-out infinite;
  }
  .agent-status-dots {
    display: inline;
  }
  .agent-status-dots span {
    animation: dot-appear 1.4s ease-in-out infinite;
    opacity: 0;
  }
  .agent-status-dots span:nth-child(1) { animation-delay: 0s; }
  .agent-status-dots span:nth-child(2) { animation-delay: 0.2s; }
  .agent-status-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dot-appear {
    0%, 80%, 100% { opacity: 0; }
    40%           { opacity: 1; }
  }
  .agent-status-meta {
    font-size: 11px; color: #48484a;
    font-variant-numeric: tabular-nums;
    margin-left: auto;
  }

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
  .action-ind svg { margin-right: 0 !important; }

  .action-label {
    flex: 1; font-size: 12.5px;
    color: #98989d; font-weight: 500;
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; min-width: 0;
  }
  .action-meta {
    font-size: 11px; color: rgba(255,255,255,0.18);
    font-variant-numeric: tabular-nums;
    white-space: nowrap; flex-shrink: 0;
  }
  .action-meta-ok { color: #30d158; font-weight: 600; margin-right: 4px; }
  .action-meta-err { color: #ff453a; font-weight: 600; margin-right: 4px; }
  /* Inline spinner shown in action-meta while running */
  .action-meta-spinner {
    display: inline-block; width: 10px; height: 10px;
    border: 1.5px solid rgba(255,255,255,0.08);
    border-top-color: rgba(255,255,255,0.35);
    border-radius: 50%; animation: spin 0.7s linear infinite;
    vertical-align: -1px; margin-right: 5px;
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
  .action-out-wrap { position: relative; margin-top: 6px; }
  .action-out {
    padding: 8px 12px;
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 8px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 11px; line-height: 1.45; color: #8e8e93;
    white-space: pre; overflow-x: auto;
    position: relative;
  }
  .action-out-wrap.collapsed .action-out {
    max-height: 120px; overflow: hidden; cursor: pointer;
  }
  .action-out-wrap.collapsed:hover + .action-out-toggle {
    color: rgba(255,255,255,0.5);
  }
  /* Fade gradient overlay when collapsed */
  .action-out-wrap.collapsed::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0;
    height: 48px; border-radius: 0 0 8px 8px; pointer-events: none;
    background: linear-gradient(
      rgba(0,0,0,0) 0%,
      rgba(0,0,0,0.18) 50%,
      rgba(0,0,0,0.30) 100%
    );
  }
  .action-out-wrap.expanded .action-out {
    max-height: none; overflow-y: auto;
  }
  .action-out::-webkit-scrollbar { width: 3px; height: 3px; }
  .action-out::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
  .action-out.e { color: rgba(255,69,58,0.75); }
  .sandbox-hint {
    display: flex; align-items: flex-start; gap: 7px;
    margin-top: 8px; padding: 8px 11px;
    background: rgba(255,159,10,0.06);
    border: 1px solid rgba(255,159,10,0.12);
    border-radius: 8px;
    font-size: 11px; line-height: 1.45;
    color: rgba(255,159,10,0.8);
  }
  .sandbox-hint-icon {
    flex-shrink: 0; margin-top: 1px;
    font-size: 12px; line-height: 1;
  }
  /* Toggle button — always a sibling below .action-out-wrap */
  .action-out-toggle {
    display: block; width: 100%; margin-top: 2px;
    padding: 4px 12px;
    background: none; border: none; cursor: pointer;
    font-size: 10.5px; color: rgba(255,255,255,0.2);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    font-weight: 500; text-align: left;
    transition: color 0.15s;
  }
  .action-out-toggle:hover { color: rgba(255,255,255,0.5); }

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

  /* ═══════════ Final script divider ═══════════ */
  .final-divider {
    padding: 16px 16px 6px;
    display: flex; align-items: center; gap: 10px;
  }
  .final-divider-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(94,92,230,0.25), transparent);
  }
  .final-divider-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(94,92,230,0.6);
    white-space: nowrap;
  }

  /* Final script action row — highlighted variant */
  .action.final-script .action-row {
    background: rgba(94,92,230,0.06);
    border-color: rgba(94,92,230,0.15);
  }
  .action.final-script .action-row:hover {
    background: rgba(94,92,230,0.09);
  }
  .action.final-script .action-label {
    color: #c4c2f0; font-weight: 600;
  }

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
    padding: 14px 14px; border-radius: 10px;
  }
  .terminal-card.success {
    background: rgba(48,209,88,0.04);
    border: 1px solid rgba(48,209,88,0.1);
  }
  .terminal-card.fail {
    background: rgba(255,69,58,0.04);
    border: 1px solid rgba(255,69,58,0.1);
  }
  .terminal-row {
    display: flex; align-items: center; gap: 10px;
  }
  .terminal-icon {
    width: 22px; height: 22px; border-radius: 50%;
    display: inline-grid; place-items: center;
    font-size: 11px; font-weight: 600; flex-shrink: 0;
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
  .plan { padding: 3px 16px; }
  .plan-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; overflow: hidden;
  }
  .plan-header {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 8px; cursor: pointer;
    user-select: none; -webkit-user-select: none;
    transition: background 0.15s;
  }
  .plan-header:hover { background: rgba(255,255,255,0.025); }
  .plan-icon {
    width: 16px; height: 16px; flex-shrink: 0;
    display: grid; place-items: center;
    color: #8e8e93;
  }
  .plan-icon svg { width: 14px; height: 14px; color: #64d2ff; }
  .plan-title {
    flex: 1; font-size: 12.5px; font-weight: 500;
    color: #98989d;
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
    padding: 4px 8px 10px; font-size: 12px; color: #8e8e93; line-height: 1.5;
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
  @keyframes char-wave {
    0%, 100% { opacity: 0.4; }
    50%      { opacity: 1; }
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

  /* ═══════════ Page map card (show_page) ═══════════ */
  .pagemap { padding: 4px 16px; }
  .pagemap-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px; overflow: hidden;
  }
  .pagemap-header {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 12px;
    font-size: 12px; font-weight: 500; color: #98989d;
    cursor: pointer; user-select: none; -webkit-user-select: none;
  }
  .pagemap-chev {
    font-size: 11px; color: rgba(255,255,255,0.1);
    transition: transform 0.2s; flex-shrink: 0;
  }
  .pagemap-card.open .pagemap-chev { transform: rotate(90deg); }
  .pagemap-sections-body {
    max-height: 0; overflow: hidden; opacity: 0;
    transition: max-height 0.35s cubic-bezier(0.4,0,0.2,1),
                opacity 0.2s ease;
  }
  .pagemap-card.open .pagemap-sections-body {
    max-height: 2000px; opacity: 1;
  }
  .pagemap-section.hidden-section { display: none; }
  .pagemap-card.show-all .pagemap-section.hidden-section { display: block; }
  .pagemap-more {
    display: block; width: 100%; padding: 5px 0; margin: 0;
    background: none; border: none; cursor: pointer;
    font-size: 10.5px; color: rgba(255,255,255,0.2);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    font-weight: 500; text-align: center;
    transition: color 0.15s;
  }
  .pagemap-more:hover { color: rgba(255,255,255,0.5); }
  .pagemap-card.show-all .pagemap-more { display: none; }
  .pagemap-icon {
    width: 16px; height: 16px; flex-shrink: 0;
    display: grid; place-items: center;
  }
  .pagemap-url {
    flex: 1; font-size: 10.5px; color: rgba(255,255,255,0.2);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }
  .pagemap-count {
    font-size: 10px; color: rgba(255,255,255,0.2);
    font-weight: 500; flex-shrink: 0;
  }
  .pagemap-sections { padding: 0 4px 4px; }
  .pagemap-section {
    border-radius: 8px;
    transition: background 0.15s;
  }
  .pagemap-section:hover { background: rgba(255,255,255,0.02); }
  .pagemap-section-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 8px; cursor: pointer; user-select: none;
    -webkit-user-select: none;
  }
  .pagemap-section-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  }
  .pagemap-section-dot.role-navigation { background: #64d2ff; }
  .pagemap-section-dot.role-content    { background: #30d158; }
  .pagemap-section-dot.role-form       { background: #ff9f0a; }
  .pagemap-section-dot.role-header     { background: #bf5af2; }
  .pagemap-section-dot.role-footer     { background: #8e8e93; }
  .pagemap-section-dot.role-other      { background: #636366; }
  .pagemap-section-id {
    font-size: 11px; font-weight: 600; color: #d1d1d6;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 140px; flex-shrink: 0;
  }
  .pagemap-section-role {
    flex: 1; font-size: 10.5px; color: rgba(255,255,255,0.25);
  }
  .pagemap-section-badge {
    font-size: 9.5px; padding: 1px 6px; border-radius: 8px;
    background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.25);
    flex-shrink: 0;
  }
  .pagemap-section-chev {
    font-size: 11px; color: rgba(255,255,255,0.1);
    transition: transform 0.2s; flex-shrink: 0;
  }
  .pagemap-section.open .pagemap-section-chev { transform: rotate(90deg); }
  .pagemap-section-content {
    max-height: 0; overflow: hidden; opacity: 0;
    transition: max-height 0.3s cubic-bezier(0.4,0,0.2,1),
                opacity 0.2s ease, padding 0.2s ease;
    padding: 0 8px;
  }
  .pagemap-section.open .pagemap-section-content {
    max-height: 300px; overflow-y: auto; opacity: 1;
    padding: 0 8px 8px;
  }
  .pagemap-section-content::-webkit-scrollbar { width: 3px; }
  .pagemap-section-content::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.06); border-radius: 3px;
  }
  .pagemap-section-pre {
    margin: 0; padding: 8px 10px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 6px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 10.5px; line-height: 1.45; color: #8e8e93;
    white-space: pre-wrap; word-break: break-word;
  }

  /* ═══════════ Zoom viewer (zoom_section) ═══════════ */
  .zoomview { padding: 4px 16px; }
  .zoomview-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px; overflow: hidden;
  }
  .zoomview-header {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 12px;
    font-size: 12px; font-weight: 500; color: #98989d;
    cursor: pointer; user-select: none; -webkit-user-select: none;
  }
  .zoomview-icon {
    width: 16px; height: 16px; flex-shrink: 0;
    display: grid; place-items: center;
  }
  .zoomview-ids {
    flex: 1; font-size: 11px; color: rgba(255,255,255,0.18); font-weight: 500;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .zoomview-chev {
    font-size: 11px; color: rgba(255,255,255,0.1);
    transition: transform 0.2s; flex-shrink: 0;
  }
  .zoomview-card.open .zoomview-chev { transform: rotate(90deg); }
  .zoomview-body {
    max-height: 0; overflow: hidden; opacity: 0;
    transition: max-height 0.3s cubic-bezier(0.4,0,0.2,1),
                opacity 0.2s ease;
  }
  .zoomview-card.open .zoomview-body {
    max-height: 10000px; overflow-y: auto; opacity: 1;
  }
  .zoomview-body::-webkit-scrollbar { width: 3px; }
  .zoomview-body::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.06); border-radius: 3px;
  }
  .zoomview-pre {
    margin: 0; padding: 10px 12px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 10.5px; line-height: 1.5; color: #8e8e93;
    white-space: pre; overflow-x: auto;
    border-top: 1px solid rgba(255,255,255,0.03);
  }
  .zoomview-pre::-webkit-scrollbar { height: 3px; }
  .zoomview-pre::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 3px; }
  /* HTML syntax highlighting */
  .zoomview-pre .ht { color: #ff6482; }
  .zoomview-pre .ha { color: #ff9f0a; }
  .zoomview-pre .hv { color: #6cd68e; }
  .zoomview-pre .hc { color: rgba(255,255,255,0.2); font-style: italic; }
  .zoomview-pre .hp { color: #8e8e93; }

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

  /* ═══════════ JSON Results Viewer ═══════════ */
  .results-section {
    padding: 0; margin: 0;
    animation: bootIn 0.3s ease;
  }

  /* Sticky header block — contains title row and search row */
  .results-header {
    position: sticky; top: -6px; z-index: 5;
    background: rgba(22,22,24,0.97);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.04);
    padding: 12px 16px 10px;
    display: flex; flex-direction: column; gap: 8px;
  }

  /* Row 1: title + toggle + copy — all in one line */
  .results-header-row1 {
    display: flex; align-items: center; gap: 12px;
  }
  .results-title {
    font-size: 12px; font-weight: 600;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.02em; text-transform: uppercase;
    white-space: nowrap;
  }
  .results-title-count {
    font-weight: 500;
    color: rgba(255,255,255,0.3);
    text-transform: none;
    letter-spacing: 0;
  }
  .results-copy-btn {
    background: none; border: none; cursor: pointer;
    color: rgba(255,255,255,0.25);
    padding: 3px; margin-left: auto;
    line-height: 0;
    transition: color 0.15s ease;
    flex-shrink: 0;
    display: flex; align-items: center;
  }
  .results-copy-btn:hover { color: rgba(255,255,255,0.6); }
  .results-copy-btn.copied { color: #30d158; }
  .results-copy-btn svg { width: 14px; height: 14px; }

  /* Row 2: search + breadcrumb */
  .results-header-row2 {
    display: flex; flex-direction: column; gap: 4px;
  }

  /* Breadcrumb */
  .jt-breadcrumb {
    display: flex; align-items: center; gap: 0;
    font-size: 11px; color: #636366;
    overflow-x: auto; white-space: nowrap;
    scrollbar-width: none;
    padding: 0;
  }
  .jt-breadcrumb::-webkit-scrollbar { display: none; }
  .jt-breadcrumb-seg {
    color: #8e8e93; cursor: pointer;
    padding: 1px 3px; border-radius: 3px;
    transition: background 0.12s ease;
  }
  .jt-breadcrumb-seg:hover {
    background: rgba(255,255,255,0.06);
    color: #d1d1d6;
  }
  .jt-breadcrumb-sep {
    color: rgba(255,255,255,0.15);
    padding: 0 3px; user-select: none;
  }

  /* Search input (shared by both views) */
  .results-search-wrap {
    position: relative;
  }
  .results-search-input {
    width: 100%; padding: 6px 10px 6px 28px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 7px; color: #e5e5e7;
    font-size: 11.5px; font-family: inherit;
    outline: none;
    transition: border-color 0.15s ease;
  }
  .results-search-input::placeholder { color: rgba(255,255,255,0.2); }
  .results-search-input:focus {
    border-color: rgba(94,92,230,0.4);
  }
  .results-search-icon {
    position: absolute; left: 8px; top: 50%; transform: translateY(-50%);
    font-size: 11px; color: rgba(255,255,255,0.2);
    pointer-events: none;
  }

  /* Tree */
  .results-tree {
    padding: 4px 0 8px 12px;
    overflow-x: auto;
  }
  .results-tree::-webkit-scrollbar { height: 4px; }
  .results-tree::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
  .results-tree::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }
  .jt-row {
    display: flex; align-items: flex-start;
    padding: 2px 12px 2px 4px;
    min-height: 24px; line-height: 20px;
    position: relative;
    transition: background 0.08s ease;
    white-space: nowrap;
  }
  .jt-row:hover {
    background: rgba(255,255,255,0.025);
  }
  .jt-row.jt-zebra {
    border-top: 1px solid rgba(255,255,255,0.03);
  }
  .jt-row.jt-hidden { display: none; }
  .jt-children { display: none; }
  .jt-children.jt-open { display: block; }

  /* Indentation guides */
  .jt-indent {
    display: inline-block; width: 20px; flex-shrink: 0;
    position: relative; align-self: stretch;
  }
  .jt-indent::after {
    content: ''; position: absolute;
    left: 9px; top: 0; bottom: 0;
    border-left: 1px solid rgba(255,255,255,0.04);
  }

  /* Toggle triangle */
  .jt-toggle {
    width: 16px; height: 20px; flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
    cursor: pointer; color: rgba(255,255,255,0.25);
    transition: transform 0.15s cubic-bezier(0.4,0,0.2,1),
                color 0.12s ease;
    font-size: 9px; user-select: none;
  }
  .jt-toggle:hover { color: rgba(255,255,255,0.5); }
  .jt-toggle.jt-expanded { transform: rotate(90deg); }
  .jt-toggle-placeholder {
    width: 16px; flex-shrink: 0;
  }

  /* Key and value — VS Code dark theme JSON colors */
  .jt-key {
    color: #9CDCFE;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; white-space: nowrap;
  }
  .jt-colon {
    color: #D4D4D4;
    margin: 0 4px 0 0;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
  }
  .jt-punct {
    color: #D4D4D4;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
  }
  .jt-comma {
    color: #D4D4D4;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
  }
  .jt-val {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; white-space: nowrap;
  }
  .jt-val.jt-string { color: #CE9178; }
  .jt-val.jt-number { color: #B5CEA8; }
  .jt-val.jt-boolean { color: #569CD6; }
  .jt-val.jt-null { color: #569CD6; }

  /* Gutter index for array items */
  .jt-gutter {
    color: rgba(255,255,255,0.18);
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 10px; min-width: 20px;
    text-align: right; padding-right: 6px;
    user-select: none; flex-shrink: 0;
    line-height: 20px;
  }

  /* Collapsed badge */
  .jt-badge {
    display: inline-block;
    font-size: 10px; font-weight: 500;
    padding: 1px 6px; margin-left: 4px;
    border-radius: 4px;
    background: rgba(255,255,255,0.05);
    color: #636366;
    font-family: inherit;
  }
  .jt-type-hint {
    color: #D4D4D4; font-size: 12px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }

  /* Copy button */
  .jt-copy {
    opacity: 0; margin-left: auto; padding-left: 8px;
    flex-shrink: 0; cursor: pointer;
    font-size: 11px; color: rgba(255,255,255,0.25);
    transition: opacity 0.12s ease, color 0.12s ease;
    white-space: nowrap; user-select: none;
    display: flex; align-items: center; height: 20px;
  }
  .jt-row:hover .jt-copy { opacity: 0.5; }
  .jt-copy:hover { opacity: 1 !important; color: rgba(255,255,255,0.7); }
  .jt-copy.copied { color: #30d158; opacity: 1 !important; }

  /* Search highlight */
  .jt-search-match {
    background: rgba(255,214,10,0.2);
    border-radius: 2px; padding: 0 1px;
  }

  /* Long string truncation */
  .jt-long-toggle {
    color: #5e5ce6; cursor: pointer;
    font-size: 11px; margin-left: 4px;
    font-family: inherit; flex-shrink: 0;
  }
  .jt-long-toggle:hover { text-decoration: underline; }

  /* Show more for large arrays */
  .jt-show-more {
    padding: 4px 16px 4px 0;
    display: flex; align-items: center;
  }
  .jt-show-more-btn {
    font-size: 11px; font-weight: 400;
    color: rgba(255,255,255,0.3); cursor: pointer;
    padding: 2px 0; border-radius: 0;
    background: none;
    border: none; font-family: inherit;
    transition: color 0.15s;
  }
  .jt-show-more-btn::before {
    content: '\203A\00a0';
    font-size: 13px; font-weight: 300;
  }
  .jt-show-more-btn:hover {
    color: rgba(255,255,255,0.55);
  }

  /* ═══════════ View toggle tabs ═══════════ */
  .results-tabs {
    display: flex; gap: 1px;
    padding: 2px;
    background: rgba(255,255,255,0.04);
    border-radius: 5px;
    flex-shrink: 0;
  }
  .results-tab {
    padding: 2px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 500;
    color: rgba(255,255,255,0.35);
    cursor: pointer; border: none;
    background: transparent;
    font-family: inherit;
    transition: all 0.15s ease;
    line-height: 1.3;
    letter-spacing: 0.02em;
  }
  .results-tab:hover {
    color: rgba(255,255,255,0.55);
  }
  .results-tab.active {
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.85);
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
  }

  /* ═══════════ Table view ═══════════ */
  .results-table-wrap {
    overflow-x: auto; overflow-y: visible;
    padding: 0 0 8px;
    display: none;
  }
  .results-table-wrap.active { display: block; }
  .results-table-wrap::-webkit-scrollbar { height: 5px; }
  .results-table-wrap::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08); border-radius: 4px;
  }
  .results-table-wrap::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.14);
  }

  .rt {
    width: 100%; border-collapse: separate;
    border-spacing: 0;
    font-size: 12px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
  }

  /* Header */
  .rt thead { position: sticky; top: 0; z-index: 3; }
  .rt th {
    padding: 8px 14px;
    text-align: left; white-space: nowrap;
    font-weight: 600; font-size: 11px;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(22,22,24,0.97);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: relative;
    user-select: none;
  }
  .rt th:first-child { padding-left: 16px; }
  .rt th .rt-row-num {
    color: rgba(255,255,255,0.2);
    font-weight: 500; font-size: 10px;
    text-transform: none; letter-spacing: 0;
  }

  /* Rows */
  .rt td {
    padding: 6px 14px;
    color: #e5e5e7;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    max-width: 320px;
    overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: top;
    transition: background 0.08s ease;
  }
  .rt td:first-child { padding-left: 16px; }
  .rt tr:hover td {
    background: rgba(255,255,255,0.025);
  }
  .rt tbody tr:nth-child(even) td {
    background: rgba(255,255,255,0.012);
  }
  .rt tbody tr:nth-child(even):hover td {
    background: rgba(255,255,255,0.035);
  }

  /* Row number column */
  .rt .rt-num {
    color: rgba(255,255,255,0.15);
    font-size: 10px;
    text-align: right;
    padding-right: 10px;
    width: 36px; min-width: 36px;
    user-select: none;
  }

  /* Typed cell values */
  .rt .rt-string { color: #CE9178; }
  .rt .rt-number { color: #B5CEA8; }
  .rt .rt-boolean { color: #569CD6; }
  .rt .rt-null { color: rgba(255,255,255,0.2); font-style: italic; }
  .rt .rt-array {
    color: #8e8e93; font-size: 11px;
  }
  .rt .rt-object {
    color: #8e8e93; font-size: 11px;
  }

  /* Cell tooltip on hover for truncated values */
  .rt td[title] { cursor: default; }

  /* Show more row */
  .rt .rt-show-more td {
    text-align: left;
    border-bottom: none;
    padding: 10px 14px;
    cursor: pointer;
  }

  /* Empty state */
  .rt-empty {
    padding: 24px 16px; text-align: center;
    color: rgba(255,255,255,0.25); font-size: 12px;
  }

  /* Footer with buttons */
  .results-footer {
    padding: 12px 16px;
    background: rgba(22,22,24,0.95);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex; flex-direction: column; gap: 8px;
    flex-shrink: 0;
  }
  .btn-expand {
    width: 100%; padding: 8px 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    color: #d1d1d6; font-size: 12px; font-weight: 500;
    font-family: inherit; cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-expand:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.12);
  }
  .btn-expand.active {
    background: rgba(255,255,255,0.08);
  }
  .btn-finish {
    width: 100%; padding: 10px 24px;
    background: linear-gradient(135deg, #5e5ce6, #bf5af2);
    border: none; border-radius: 10px;
    color: #fff; font-size: 13px; font-weight: 600;
    font-family: inherit; cursor: pointer;
    transition: filter 0.15s ease, transform 0.1s ease;
    letter-spacing: 0.01em;
  }
  .btn-finish:hover { filter: brightness(1.12); }
  .btn-finish:active { transform: scale(0.98); filter: brightness(0.95); }
  .btn-finish.closing {
    pointer-events: none; opacity: 0.7;
  }
  .btn-finish.closing::before {
    content: ''; display: inline-block;
    width: 12px; height: 12px; margin-right: 8px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%; animation: spin 0.7s linear infinite;
    vertical-align: -2px;
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
  let _programmaticScroll = false;
  let _scrollEndTimer = 0;

  /* Sentinel element at the very bottom of the feed — used as the
     scroll target so we always reach the true end. */
  const _sentinel = document.createElement('div');
  _sentinel.style.cssText = 'height:1px;width:100%;flex-shrink:0;';
  feed.appendChild(_sentinel);

  /* Detect whether user scrolled away from the bottom.
     During a programmatic smooth scroll we ignore scroll events
     because the sentinel hasn't been reached yet — without this
     guard, the observer would set _userNearBottom=false mid-
     animation and kill auto-scroll. */
  feed.addEventListener('scroll', () => {
    if (_programmaticScroll) return;
    const gap = feed.scrollHeight - feed.scrollTop - feed.clientHeight;
    _userNearBottom = gap < 60;
  }, { passive: true });

  function scrollDown() {
    if (_replaying) return;
    if (!_userNearBottom) return;
    _programmaticScroll = true;
    clearTimeout(_scrollEndTimer);
    feed.scrollTo({ top: feed.scrollHeight, behavior: 'smooth' });
    /* Release the guard after the smooth scroll settles. */
    _scrollEndTimer = setTimeout(() => { _programmaticScroll = false; }, 400);
  }
  /* Insert content before the sentinel so it stays at the very bottom. */
  function feedAppend(el) { feed.insertBefore(el, _sentinel); }
  function trim() {
    /* +1 accounts for the sentinel element at the end. */
    while (feed.children.length > 121) feed.removeChild(feed.firstChild);
  }

  const _OUT_COLLAPSE_LINES = 8;
  function makeActionOut(text, isError) {
    const frag = document.createDocumentFragment();
    const wrap = document.createElement('div');
    wrap.className = 'action-out-wrap';
    const o = document.createElement('div');
    o.className = 'action-out' + (isError ? ' e' : '');
    o.textContent = text;
    wrap.appendChild(o);
    frag.appendChild(wrap);

    const lines = text.split('\n').length;
    if (lines > _OUT_COLLAPSE_LINES) {
      wrap.classList.add('collapsed');
      const btn = document.createElement('button');
      btn.className = 'action-out-toggle';
      btn.textContent = 'Show all ' + lines + ' lines';
      frag.appendChild(btn);

      function expand() {
        wrap.classList.remove('collapsed');
        wrap.classList.add('expanded');
        btn.textContent = 'Show less';
      }
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (wrap.classList.contains('collapsed')) {
          expand();
        } else {
          wrap.classList.remove('expanded');
          wrap.classList.add('collapsed');
          btn.textContent = 'Show all ' + lines + ' lines';
        }
      });
      /* Click anywhere on collapsed output expands it. */
      o.addEventListener('click', () => {
        if (wrap.classList.contains('collapsed')) expand();
      });
    }
    return frag;
  }

  /* ═══════════ JSON Results Viewer ═══════════ */
  const INITIAL_BATCH = 10;
  const BATCH_SIZE = 50;
  const MAX_STRING_LEN = 500;
  let _jtBreadcrumbPath = [];

  function copyToClipboard(text, el) {
    const write = () => navigator.clipboard.writeText(text);
    write().catch(() => {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.cssText = 'position:fixed;left:-9999px';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy'); ta.remove();
    });
    if (el) {
      const prev = el.innerHTML;
      el.classList.add('copied');
      // For SVG icon buttons, swap to checkmark SVG; for text, swap text
      if (el.querySelector('svg')) {
        el.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px"><polyline points="20 6 9 17 4 12"/></svg>';
      } else {
        el.textContent = '\u2713';
      }
      setTimeout(() => { el.innerHTML = prev; el.classList.remove('copied'); }, 1200);
    }
  }

  function jtCount(val) {
    if (Array.isArray(val)) return val.length + ' item' + (val.length !== 1 ? 's' : '');
    if (val && typeof val === 'object') {
      const n = Object.keys(val).length;
      return n + ' key' + (n !== 1 ? 's' : '');
    }
    return '';
  }

  function jtTypeClass(val) {
    if (val === null || val === undefined) return 'jt-null';
    if (typeof val === 'string') return 'jt-string';
    if (typeof val === 'number') return 'jt-number';
    if (typeof val === 'boolean') return 'jt-boolean';
    return '';
  }

  function jtDisplayVal(val) {
    if (val === null) return 'null';
    if (val === undefined) return 'undefined';
    if (typeof val === 'string') {
      if (val.length > MAX_STRING_LEN) return '"' + esc(val.slice(0, MAX_STRING_LEN)) + '\u2026"';
      return '"' + esc(val) + '"';
    }
    if (typeof val === 'boolean') return val ? 'true' : 'false';
    return String(val);
  }

  function jtIsComplex(val) {
    return val !== null && typeof val === 'object';
  }

  function updateBreadcrumb(bcEl, path) {
    _jtBreadcrumbPath = path;
    if (!bcEl) return;
    bcEl.innerHTML = '';
    path.forEach((seg, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'jt-breadcrumb-sep';
        sep.textContent = '\u203A';
        bcEl.appendChild(sep);
      }
      const s = document.createElement('span');
      s.className = 'jt-breadcrumb-seg';
      s.textContent = seg;
      bcEl.appendChild(s);
    });
    bcEl.scrollLeft = bcEl.scrollWidth;
  }

  function renderNode(value, parent, key, depth, path, isArrayItem, arrayIndex, bcEl, isLastSibling) {
    const isComplex = jtIsComplex(value);
    const isArr = Array.isArray(value);
    const showComma = !isLastSibling;

    const row = document.createElement('div');
    row.className = 'jt-row';
    if (depth === 1 && isArrayItem && arrayIndex % 2 === 0) row.classList.add('jt-zebra');
    row._jtPath = path;
    row._jtValue = value;

    // Gutter number for array items
    if (isArrayItem) {
      const gutter = document.createElement('span');
      gutter.className = 'jt-gutter';
      gutter.textContent = String(arrayIndex + 1);
      row.appendChild(gutter);
    }

    // Indentation
    for (let i = 0; i < depth; i++) {
      const indent = document.createElement('span');
      indent.className = 'jt-indent';
      row.appendChild(indent);
    }

    if (isComplex) {
      // Toggle
      const toggle = document.createElement('span');
      toggle.className = 'jt-toggle';
      toggle.textContent = '\u25B6';
      row.appendChild(toggle);

      // Key (quoted for object properties, omitted for array items)
      if (key !== null && key !== undefined && !isArrayItem) {
        const k = document.createElement('span');
        k.className = 'jt-key';
        k.textContent = '"' + key + '"';
        row.appendChild(k);
        const colon = document.createElement('span');
        colon.className = 'jt-colon';
        colon.textContent = ': ';
        row.appendChild(colon);
      }

      // Opening bracket
      const hint = document.createElement('span');
      hint.className = 'jt-punct';
      hint.textContent = isArr ? '[' : '{';
      row.appendChild(hint);

      // Badge — always visible to show total count
      const badge = document.createElement('span');
      badge.className = 'jt-badge';
      badge.textContent = jtCount(value);
      row.appendChild(badge);

      // Closing bracket + comma (shown when collapsed)
      const closeHint = document.createElement('span');
      closeHint.className = 'jt-punct';
      closeHint.textContent = (isArr ? ']' : '}') + (showComma ? ',' : '');
      row.appendChild(closeHint);

      // Copy button
      const cp = document.createElement('span');
      cp.className = 'jt-copy';
      cp.textContent = 'Copy';
      cp.addEventListener('click', (e) => {
        e.stopPropagation();
        copyToClipboard(JSON.stringify(value, null, 2), cp);
      });
      row.appendChild(cp);

      parent.appendChild(row);

      // Children container
      const children = document.createElement('div');
      children.className = 'jt-children';
      children._jtData = value;
      children._jtDepth = depth + 1;
      children._jtPath = path;
      children._jtRendered = false;
      children._jtBcEl = bcEl;
      parent.appendChild(children);

      // Closing bracket row (shown when expanded)
      const closeRow = document.createElement('div');
      closeRow.className = 'jt-row';
      closeRow.style.display = 'none';
      if (isArrayItem) {
        const g = document.createElement('span');
        g.className = 'jt-gutter'; g.textContent = '';
        closeRow.appendChild(g);
      }
      for (let i = 0; i < depth; i++) {
        const indent = document.createElement('span');
        indent.className = 'jt-indent';
        closeRow.appendChild(indent);
      }
      const closePlaceholder = document.createElement('span');
      closePlaceholder.className = 'jt-toggle-placeholder';
      closeRow.appendChild(closePlaceholder);
      const closeBracket = document.createElement('span');
      closeBracket.className = 'jt-punct';
      closeBracket.textContent = (isArr ? ']' : '}') + (showComma ? ',' : '');
      closeRow.appendChild(closeBracket);
      parent.appendChild(closeRow);

      const autoExpand = depth < 2;

      function expandNode() {
        if (!children._jtRendered) {
          children._jtRendered = true;
          const entries = isArr ? value.map((v, i) => [i, v]) : Object.entries(value);
          const total = entries.length;
          let rendered = 0;

          function renderBatch(start, count) {
            const end = Math.min(start + count, total);
            for (let i = start; i < end; i++) {
              const [ek, ev] = entries[i];
              const childPath = path.concat(isArr ? '[' + ek + ']' : ek);
              const isLast = (i === total - 1);
              renderNode(ev, children, ek, depth + 1, childPath, isArr, i, bcEl, isLast);
            }
            rendered = end;

            // Show more button for large collections
            if (end < total) {
              const existing = children.querySelector('.jt-show-more');
              if (existing) existing.remove();
              const more = document.createElement('div');
              more.className = 'jt-show-more';
              if (isArr) {
                const g = document.createElement('span');
                g.className = 'jt-gutter'; g.textContent = '';
                more.appendChild(g);
              }
              for (let i = 0; i <= depth; i++) {
                const indent = document.createElement('span');
                indent.className = 'jt-indent';
                more.appendChild(indent);
              }
              const btn = document.createElement('button');
              btn.className = 'jt-show-more-btn';
              btn.textContent = 'Show ' + Math.min(BATCH_SIZE, total - end) + ' more of ' + (total - end) + ' remaining';
              function loadMoreJson() { more.remove(); renderBatch(end, BATCH_SIZE); }
              btn.addEventListener('click', loadMoreJson);
              more.style.cursor = 'pointer';
              more.addEventListener('click', loadMoreJson);
              more.appendChild(btn);
              children.appendChild(more);
            }
          }

          renderBatch(0, INITIAL_BATCH);
        }
        toggle.classList.add('jt-expanded');
        children.classList.add('jt-open');
        closeHint.style.display = 'none';
        closeRow.style.display = '';
        updateBreadcrumb(bcEl, path);
      }

      function collapseNode() {
        toggle.classList.remove('jt-expanded');
        children.classList.remove('jt-open');
        closeHint.style.display = '';
        closeRow.style.display = 'none';
      }

      toggle.addEventListener('click', () => {
        if (children.classList.contains('jt-open')) collapseNode();
        else expandNode();
      });
      row.addEventListener('dblclick', () => {
        if (children.classList.contains('jt-open')) collapseNode();
        else expandNode();
      });

      if (autoExpand) expandNode();

    } else {
      // Primitive leaf node
      const placeholder = document.createElement('span');
      placeholder.className = 'jt-toggle-placeholder';
      row.appendChild(placeholder);

      // Key (quoted for object properties, omitted for array items)
      if (key !== null && key !== undefined && !isArrayItem) {
        const k = document.createElement('span');
        k.className = 'jt-key';
        k.textContent = '"' + key + '"';
        row.appendChild(k);
        const colon = document.createElement('span');
        colon.className = 'jt-colon';
        colon.textContent = ': ';
        row.appendChild(colon);
      }

      const v = document.createElement('span');
      v.className = 'jt-val ' + jtTypeClass(value);
      v.textContent = jtDisplayVal(value);
      v._jtRawValue = value;
      row.appendChild(v);

      // Comma
      if (showComma) {
        const comma = document.createElement('span');
        comma.className = 'jt-comma';
        comma.textContent = ',';
        row.appendChild(comma);
      }

      // Long string toggle
      if (typeof value === 'string' && value.length > MAX_STRING_LEN) {
        let showFull = false;
        const toggle = document.createElement('span');
        toggle.className = 'jt-long-toggle';
        toggle.textContent = 'Show full';
        toggle.addEventListener('click', () => {
          showFull = !showFull;
          v.textContent = showFull
            ? '"' + value + '"'
            : jtDisplayVal(value);
          toggle.textContent = showFull ? 'Collapse' : 'Show full';
        });
        row.appendChild(toggle);
      }

      // Copy button
      const cp = document.createElement('span');
      cp.className = 'jt-copy';
      cp.textContent = 'Copy';
      cp.addEventListener('click', (e) => {
        e.stopPropagation();
        const raw = typeof value === 'string' ? value : JSON.stringify(value);
        copyToClipboard(raw, cp);
      });
      row.appendChild(cp);

      parent.appendChild(row);
    }
  }

  function filterTree(treeEl, query) {
    const rows = treeEl.querySelectorAll('.jt-row');
    const q = query.toLowerCase();

    // Remove old highlights
    treeEl.querySelectorAll('.jt-search-match').forEach(el => {
      el.replaceWith(el.textContent);
    });

    if (!q) {
      rows.forEach(r => r.classList.remove('jt-hidden'));
      return;
    }

    const matched = new Set();

    rows.forEach(row => {
      const keyEl = row.querySelector('.jt-key');
      const valEl = row.querySelector('.jt-val');
      const keyText = keyEl ? keyEl.textContent.toLowerCase() : '';
      const valText = valEl ? valEl.textContent.toLowerCase() : '';

      if (keyText.includes(q) || valText.includes(q)) {
        matched.add(row);
        // Walk up to mark ancestors visible
        let el = row.parentElement;
        while (el && el !== treeEl) {
          if (el.classList && el.classList.contains('jt-row')) matched.add(el);
          // Ensure parent containers are open
          if (el.classList && el.classList.contains('jt-children')) {
            el.classList.add('jt-open');
          }
          el = el.parentElement;
        }

        // Highlight matches in text
        [keyEl, valEl].forEach(target => {
          if (!target) return;
          const text = target.textContent;
          const idx = text.toLowerCase().indexOf(q);
          if (idx === -1) return;
          const before = text.slice(0, idx);
          const match = text.slice(idx, idx + q.length);
          const after = text.slice(idx + q.length);
          target.innerHTML = esc(before)
            + '<span class="jt-search-match">' + esc(match) + '</span>'
            + esc(after);
        });
      }
    });

    rows.forEach(row => {
      if (matched.has(row)) row.classList.remove('jt-hidden');
      else row.classList.add('jt-hidden');
    });
  }

  /* ═══════════ Table view renderer ═══════════ */
  const TABLE_INITIAL = 20;
  const TABLE_BATCH = 50;

  function canShowTable(data) {
    if (!Array.isArray(data) || data.length === 0) return false;
    // At least 80% of items must be objects (not arrays, not primitives)
    let objCount = 0;
    const sample = data.slice(0, Math.min(50, data.length));
    for (const item of sample) {
      if (item && typeof item === 'object' && !Array.isArray(item)) objCount++;
    }
    return objCount / sample.length >= 0.8;
  }

  function collectColumns(data) {
    // Gather all unique keys, preserving first-seen order
    const seen = new Map();
    const sample = data.slice(0, Math.min(100, data.length));
    for (const item of sample) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
      for (const key of Object.keys(item)) {
        if (!seen.has(key)) seen.set(key, 0);
        seen.set(key, seen.get(key) + 1);
      }
    }
    // Return keys that appear in at least 20% of sampled rows
    const threshold = sample.length * 0.2;
    return [...seen.entries()]
      .filter(([, count]) => count >= threshold)
      .map(([key]) => key);
  }

  function formatCell(value) {
    if (value === null || value === undefined)
      return { text: 'null', cls: 'rt-null', raw: 'null' };
    if (typeof value === 'boolean')
      return { text: String(value), cls: 'rt-boolean', raw: String(value) };
    if (typeof value === 'number')
      return { text: String(value), cls: 'rt-number', raw: String(value) };
    if (typeof value === 'string')
      return { text: value, cls: 'rt-string', raw: value };
    if (Array.isArray(value)) {
      // Array of primitives → comma-separated
      if (value.length === 0) return { text: '[]', cls: 'rt-array', raw: '[]' };
      const allPrimitive = value.every(v => v === null || typeof v !== 'object');
      if (allPrimitive) {
        const joined = value.map(v => v === null ? 'null' : typeof v === 'string' ? v : String(v)).join(', ');
        return { text: joined, cls: 'rt-array', raw: joined };
      }
      return { text: '[' + value.length + ' items]', cls: 'rt-array', raw: JSON.stringify(value) };
    }
    if (typeof value === 'object') {
      const keys = Object.keys(value);
      if (keys.length <= 3) {
        const preview = keys.map(k => k + ': ' + (typeof value[k] === 'string' ? value[k] : JSON.stringify(value[k]))).join(', ');
        if (preview.length <= 80) return { text: preview, cls: 'rt-object', raw: preview };
      }
      return { text: '{' + keys.length + ' keys}', cls: 'rt-object', raw: JSON.stringify(value) };
    }
    return { text: String(value), cls: '', raw: String(value) };
  }

  function buildTable(data, container) {
    const cols = collectColumns(data);
    if (cols.length === 0) {
      container.innerHTML = '<div class="rt-empty">No columns detected</div>';
      return;
    }

    const table = document.createElement('table');
    table.className = 'rt';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const thNum = document.createElement('th');
    thNum.innerHTML = '<span class="rt-row-num">#</span>';
    thNum.className = 'rt-num';
    headerRow.appendChild(thNum);
    for (const col of cols) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    let rendered = 0;

    function renderRows(start, count) {
      const end = Math.min(start + count, data.length);
      for (let i = start; i < end; i++) {
        const item = data[i];
        const tr = document.createElement('tr');
        tr._rtIndex = i;

        // Row number
        const tdNum = document.createElement('td');
        tdNum.className = 'rt-num';
        tdNum.textContent = String(i + 1);
        tr.appendChild(tdNum);

        for (const col of cols) {
          const td = document.createElement('td');
          const val = item && typeof item === 'object' ? item[col] : undefined;
          const formatted = formatCell(val);
          td.textContent = formatted.text;
          if (formatted.cls) td.classList.add(formatted.cls);
          // Tooltip for truncated values
          if (formatted.raw.length > 40) td.title = formatted.raw;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      rendered = end;

      // Remove old "show more" if present
      const oldMore = tbody.querySelector('.rt-show-more');
      if (oldMore) oldMore.remove();

      // Show more row
      if (end < data.length) {
        const moreRow = document.createElement('tr');
        moreRow.className = 'rt-show-more';
        const moreTd = document.createElement('td');
        moreTd.colSpan = cols.length + 1;
        const btn = document.createElement('button');
        btn.className = 'jt-show-more-btn';
        const remaining = data.length - end;
        btn.textContent = 'Show ' + Math.min(TABLE_BATCH, remaining) + ' more of ' + remaining + ' remaining';
        function loadMore() { moreRow.remove(); renderRows(end, TABLE_BATCH); }
        btn.addEventListener('click', loadMore);
        moreTd.addEventListener('click', loadMore);
        moreTd.appendChild(btn);
        moreRow.appendChild(moreTd);
        tbody.appendChild(moreRow);
      }
    }

    renderRows(0, TABLE_INITIAL);
    table.appendChild(tbody);
    container.appendChild(table);

    // Return refs for search filtering
    return { table, tbody, cols };
  }

  function filterTable(tbody, cols, query) {
    if (!tbody) return;
    const q = query.toLowerCase();
    const rows = tbody.querySelectorAll('tr:not(.rt-show-more)');
    rows.forEach(tr => {
      if (!q) { tr.style.display = ''; return; }
      const tds = tr.querySelectorAll('td:not(.rt-num)');
      let match = false;
      tds.forEach(td => {
        const text = (td.title || td.textContent || '').toLowerCase();
        if (text.includes(q)) match = true;
      });
      tr.style.display = match ? '' : 'none';
    });
  }

  function buildResultsViewer(jsonStr) {
    let data;
    try {
      data = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : jsonStr;
    } catch (e) {
      // Invalid JSON — show raw text fallback
      const d = document.createElement('div');
      d.className = 'results-section';
      d.innerHTML = '<div class="results-header"><div class="results-header-left">'
        + '<span class="results-title">Results</span></div></div>'
        + '<pre style="padding:12px 16px;font-size:11px;color:#8e8e93;overflow:auto;white-space:pre-wrap;word-break:break-word">'
        + esc(String(jsonStr)) + '</pre>';
      feedAppend(d);
      // Footer
      const footer = document.createElement('div');
      footer.className = 'results-footer';
      footer.innerHTML = '<button class="btn-expand" id="btn-expand">Expand view</button>'
        + '<button class="btn-finish" id="btn-finish">Finish</button>';
      root.appendChild(footer);
      footer.querySelector('#btn-finish').addEventListener('click', function() {
        this.classList.add('closing');
        this.textContent = 'Closing\u2026';
        if (window.__scout_dismiss) window.__scout_dismiss();
      });
      footer.querySelector('#btn-expand').addEventListener('click', function() {
        this.classList.toggle('active');
        this.textContent = this.classList.contains('active') ? 'Collapse view' : 'Expand view';
        if (window.__scout_expand) window.__scout_expand(this.classList.contains('active'));
      });
      scrollDown();
      return;
    }

    window.__scout_json_data = data;
    const tableAvailable = canShowTable(data);
    let activeView = 'json';
    let tableRefs = null;

    const section = document.createElement('div');
    section.className = 'results-section';

    // ═══════ Sticky header block ═══════
    const header = document.createElement('div');
    header.className = 'results-header';

    // Row 1: title (with inline count) + copy icon
    const row1 = document.createElement('div');
    row1.className = 'results-header-row1';
    const titleEl = document.createElement('span');
    titleEl.className = 'results-title';
    const count = jtCount(data);
    titleEl.innerHTML = 'Results' + (count
      ? ' <span class="results-title-count">(' + esc(count) + ')</span>'
      : '');
    row1.appendChild(titleEl);

    // View toggle tabs (inline, only when table is available)
    let tabJson, tabTable;
    if (tableAvailable) {
      const tabs = document.createElement('div');
      tabs.className = 'results-tabs';
      tabJson = document.createElement('button');
      tabJson.className = 'results-tab active';
      tabJson.textContent = 'JSON';
      tabTable = document.createElement('button');
      tabTable.className = 'results-tab';
      tabTable.textContent = 'Table';
      tabs.appendChild(tabJson);
      tabs.appendChild(tabTable);
      row1.appendChild(tabs);
    }

    // Copy icon (SVG clipboard)
    const copyBtn = document.createElement('button');
    copyBtn.className = 'results-copy-btn';
    copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
      + '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>'
      + '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
      + '</svg>';
    copyBtn.title = 'Copy JSON';
    copyBtn.addEventListener('click', () => {
      if (activeView === 'table') {
        const cols = tableRefs ? tableRefs.cols : [];
        let csv = cols.join(',') + '\n';
        for (const item of data) {
          if (!item || typeof item !== 'object') continue;
          csv += cols.map(c => {
            const v = item[c];
            if (v === null || v === undefined) return '';
            if (typeof v === 'string') return '"' + v.replace(/"/g, '""') + '"';
            if (typeof v === 'object') return '"' + JSON.stringify(v).replace(/"/g, '""') + '"';
            return String(v);
          }).join(',') + '\n';
        }
        copyToClipboard(csv, copyBtn);
      } else {
        copyToClipboard(JSON.stringify(data, null, 2), copyBtn);
      }
    });
    row1.appendChild(copyBtn);
    header.appendChild(row1);

    // Row 2: search
    const row2 = document.createElement('div');
    row2.className = 'results-header-row2';
    const searchWrap = document.createElement('div');
    searchWrap.className = 'results-search-wrap';
    const searchIcon = document.createElement('span');
    searchIcon.className = 'results-search-icon';
    searchIcon.textContent = '\u2315';
    const searchInput = document.createElement('input');
    searchInput.className = 'results-search-input';
    searchInput.type = 'text';
    searchInput.placeholder = 'Search keys and values\u2026';
    searchWrap.appendChild(searchIcon);
    searchWrap.appendChild(searchInput);
    row2.appendChild(searchWrap);
    header.appendChild(row2);

    section.appendChild(header);

    // ═══════ JSON tree view ═══════
    const jsonWrap = document.createElement('div');
    jsonWrap.className = 'results-tree';
    section.appendChild(jsonWrap);
    renderNode(data, jsonWrap, null, 0, ['root'], false, 0, null, true);

    // ═══════ Table view (hidden, lazy-built) ═══════
    const tableWrap = document.createElement('div');
    tableWrap.className = 'results-table-wrap';
    section.appendChild(tableWrap);

    feedAppend(section);

    // ── View switching ──
    function switchView(view) {
      activeView = view;
      if (tabJson) tabJson.classList.toggle('active', view === 'json');
      if (tabTable) tabTable.classList.toggle('active', view === 'table');
      jsonWrap.style.display = view === 'json' ? '' : 'none';
      tableWrap.classList.toggle('active', view === 'table');
      searchInput.placeholder = view === 'json'
        ? 'Search keys and values\u2026' : 'Filter rows\u2026';
      searchInput.value = '';
      copyBtn.title = view === 'table' ? 'Copy CSV' : 'Copy JSON';
      if (view === 'table' && !tableRefs) {
        tableRefs = buildTable(data, tableWrap);
      }
    }
    if (tabJson) tabJson.addEventListener('click', () => switchView('json'));
    if (tabTable) tabTable.addEventListener('click', () => switchView('table'));

    // ── Unified search ──
    let searchTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        const q = searchInput.value.trim();
        if (activeView === 'json') filterTree(jsonWrap, q);
        else if (tableRefs) filterTable(tableRefs.tbody, tableRefs.cols, q);
      }, 200);
    });

    // ── Footer ──
    const footer = document.createElement('div');
    footer.className = 'results-footer';
    footer.id = 'results-footer';

    const expandBtn = document.createElement('button');
    expandBtn.className = 'btn-expand';
    expandBtn.id = 'btn-expand';
    expandBtn.textContent = 'Expand view';
    expandBtn.addEventListener('click', () => {
      expandBtn.classList.toggle('active');
      expandBtn.textContent = expandBtn.classList.contains('active') ? 'Collapse view' : 'Expand view';
      if (window.__scout_expand) window.__scout_expand(expandBtn.classList.contains('active'));
    });
    footer.appendChild(expandBtn);

    const finishBtn = document.createElement('button');
    finishBtn.className = 'btn-finish';
    finishBtn.id = 'btn-finish';
    finishBtn.textContent = 'Finish';
    finishBtn.addEventListener('click', () => {
      finishBtn.classList.add('closing');
      finishBtn.textContent = 'Closing\u2026';
      if (window.__scout_dismiss) window.__scout_dismiss();
    });
    footer.appendChild(finishBtn);

    root.appendChild(footer);

    // Scroll to results
    requestAnimationFrame(() => {
      _userNearBottom = true;
      feed.scrollTop = feed.scrollHeight;
    });
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

  const _pySvg = '<svg viewBox="0 0 256 255" xmlns="http://www.w3.org/2000/svg" style="width:14px;height:14px;vertical-align:-2px;margin-right:7px"><defs><linearGradient id="pa" x1="12.96%" y1="12.07%" x2="79.68%" y2="78.21%"><stop offset="0%" stop-color="#387EB8"/><stop offset="100%" stop-color="#366994"/></linearGradient><linearGradient id="pb" x1="19.13%" y1="20.58%" x2="90.08%" y2="88.29%"><stop offset="0%" stop-color="#FFE052"/><stop offset="100%" stop-color="#FFC331"/></linearGradient></defs><path d="M126.9.1c-64.8 0-60.8 28.1-60.8 28.1l.1 29.2h61.9v8.7H39.2S0 61.5 0 127.1c0 65.6 34.2 63.3 34.2 63.3h20.4v-30.5s-1.1-34.2 33.6-34.2h57.9s32.6.5 32.6-31.5V32.6S183.5.1 126.9.1zm-32.2 18.8a10.5 10.5 0 1 1 0 21 10.5 10.5 0 0 1 0-21z" fill="url(#pa)"/><path d="M128.8 254.1c64.8 0 60.8-28.1 60.8-28.1l-.1-29.2h-61.9v-8.7h88.9s39.2 4.6 39.2-61 -34.2-63.3-34.2-63.3h-20.4v30.5s1.1 34.2-33.6 34.2h-57.9s-32.6-.5-32.6 31.5v61.6s-4.9 32.5 51.8 32.5zm32.2-18.8a10.5 10.5 0 1 1 0-21 10.5 10.5 0 0 1 0 21z" fill="url(#pb)"/></svg>';

  /* ── HTML syntax highlighter ─────────────────────────────────── */
  function hiHtml(code) {
    const out = [];
    const re = /(<!--[\s\S]*?-->)|(<\/?)([\w-]+)((?:\s+[\w-]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*))?)*)\s*(\/?>)|([^<]+)/g;
    let m;
    while ((m = re.exec(code)) !== null) {
      if (m[1]) { out.push('<span class="hc">' + esc(m[1]) + '</span>'); }
      else if (m[2]) {
        out.push('<span class="hp">' + esc(m[2]) + '</span>');
        out.push('<span class="ht">' + esc(m[3]) + '</span>');
        if (m[4]) {
          const attrStr = m[4];
          const attrRe = /([\w-]+)(\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*))?/g;
          let a;
          while ((a = attrRe.exec(attrStr)) !== null) {
            out.push(' <span class="ha">' + esc(a[1]) + '</span>');
            if (a[2]) {
              const eq = a[2].indexOf('=');
              const val = a[2].slice(eq + 1).trim();
              out.push('=<span class="hv">' + esc(val) + '</span>');
            }
          }
        }
        out.push('<span class="hp">' + esc(m[5]) + '</span>');
      }
      else if (m[6]) { out.push(esc(m[6])); }
    }
    return out.join('');
  }

  function roleClass(role) {
    const r = (role || '').toLowerCase();
    if (r.includes('nav')) return 'role-navigation';
    if (r.includes('content') || r.includes('main') || r.includes('article')) return 'role-content';
    if (r.includes('form') || r.includes('input') || r.includes('search')) return 'role-form';
    if (r.includes('header') || r.includes('banner')) return 'role-header';
    if (r.includes('footer')) return 'role-footer';
    return 'role-other';
  }

  /* ═══════════ State ═══════════ */
  let lastAction = null;
  let _actionTimer = null;
  function startActionTimer(metaEl, budget) {
    stopActionTimer();
    const t0 = Date.now();
    const suffix = budget ? ' / ' + budget + 's' : '';
    const spinnerEl = document.createElement('span');
    spinnerEl.className = 'action-meta-spinner';
    const timerText = document.createTextNode('');
    metaEl.textContent = '';
    metaEl.appendChild(spinnerEl);
    metaEl.appendChild(timerText);
    function tick() {
      timerText.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's' + suffix;
    }
    tick();
    _actionTimer = setInterval(tick, 100);
  }
  function stopActionTimer() {
    if (_actionTimer) { clearInterval(_actionTimer); _actionTimer = null; }
  }

  function setStatus(text, color) {
    const st = document.getElementById('status-text');
    const dot = document.getElementById('dot');
    if (st) st.textContent = text;
    if (dot) { dot.style.animation = 'none'; dot.style.background = color; }
  }

  /* ── Agent status indicator (shared by reasoning + plan gen) ── */
  function buildStatusEl(label) {
    const el = document.createElement('div');
    el.className = 'agent-status';
    /* Label text */
    const textSpan = document.createElement('span');
    textSpan.className = 'agent-status-text';
    for (let i = 0; i < label.length; i++) {
      const ch = document.createElement('span');
      ch.textContent = label[i] === ' ' ? '\u00a0' : label[i];
      ch.style.animationDelay = (i * 0.06) + 's';
      textSpan.appendChild(ch);
    }
    el.appendChild(textSpan);
    /* Animated dots */
    const dots = document.createElement('span');
    dots.className = 'agent-status-dots';
    for (let i = 0; i < 3; i++) {
      const d = document.createElement('span');
      d.textContent = '.';
      dots.appendChild(d);
    }
    el.appendChild(dots);
    /* Meta slot (elapsed or percentage) */
    const meta = document.createElement('span');
    meta.className = 'agent-status-meta';
    el.appendChild(meta);
    return { el, meta };
  }

  let _thinkingEl = null;
  let _thinkingTimer = null;
  function showThinking() {
    removeThinking();
    const { el, meta } = buildStatusEl('Reasoning');
    feedAppend(el);
    scrollDown();
    _thinkingEl = el;
    const t0 = Date.now();
    function tick() {
      meta.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
    }
    tick();
    _thinkingTimer = setInterval(tick, 100);
  }
  function removeThinking() {
    if (_thinkingTimer) { clearInterval(_thinkingTimer); _thinkingTimer = null; }
    if (_thinkingEl) { _thinkingEl.remove(); _thinkingEl = null; }
  }

  /* ═══════════ Controller ═══════════ */
  window.__scout = {
    replayEvents(events) {
      removeThinking();
      feed.innerHTML = '';
      feed.appendChild(_sentinel);
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
        feedAppend(d); scrollDown(); return;
      }

      if (t === 'thinking_start') { showThinking(); return; }
      if (t === 'thinking_end')   { removeThinking(); return; }

      if (t === 'thinking') {
        removeThinking();
        const entry = document.createElement('div');
        entry.className = 'think';
        const body = document.createElement('div');
        body.className = 'think-body';
        body.innerHTML = md(ev.text || '');
        entry.appendChild(body);
        feedAppend(entry); trim(); scrollDown();
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
        const entry = document.createElement('div');
        entry.className = 'action';
        const row = document.createElement('div');
        row.className = 'action-row';
        const ind = document.createElement('div');
        ind.className = 'action-ind';
        ind.innerHTML = _pySvg;
        const lbl = document.createElement('span');
        lbl.className = 'action-label';
        lbl.textContent = 'Python';
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
        feedAppend(entry); lastAction = entry; trim(); scrollDown();
        startActionTimer(meta, ev.timeout_budget || '');
        return;
      }

      if (t === 'tool_result') {
        stopActionTimer();
        const target = lastAction;
        if (target) {
          const meta = target.querySelector('.action-meta');
          const ok = !ev.is_error;
          const sym = ok ? '\u2713' : '\u2717';
          const cls = ok ? 'action-meta-ok' : 'action-meta-err';
          let metaHtml = '<span class="' + cls + '">' + sym + '</span> ';
          if (ev.duration_s) {
            metaHtml += ev.duration_s + 's';
            if (ev.timeout_info) {
              const m = ev.timeout_info.match(/^(\d+)s/);
              if (m) metaHtml += ' / ' + m[1] + 's';
            }
          }
          meta.innerHTML = metaHtml;
          const detail = target.querySelector('.action-detail');
          if (ev.output) {
            detail.appendChild(makeActionOut(ev.output, false));
          }
          if (ev.error) {
            detail.appendChild(makeActionOut(ev.error, true));
          }
          lastAction = null;
        }
        scrollDown(); return;
      }

      if (t === 'page_update') {
        /* Kept as a lightweight system line when no structured
           section data is provided (fallback). */
        if (!ev.section_data || !ev.section_data.length) {
          const d = document.createElement('div');
          d.className = 'sys';
          d.innerHTML = '<span class="sys-dot"></span>Page captured \u2014 '
                      + esc(String(ev.sections || '?')) + ' sections';
          feedAppend(d); scrollDown(); return;
        }
        /* Structured page map card — starts open, collapsible. */
        const d = document.createElement('div');
        d.className = 'pagemap';
        const card = document.createElement('div');
        card.className = 'pagemap-card';
        const hdr = document.createElement('div');
        hdr.className = 'pagemap-header';
        hdr.innerHTML = '<span class="pagemap-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64d2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="4" rx="1"/><rect x="14" y="10" width="7" height="11" rx="1"/><rect x="3" y="13" width="7" height="8" rx="1"/></svg></span>'
          + '<span style="flex-shrink:0">Page structure</span>'
          + '<span class="pagemap-url">' + esc(ev.url || '') + '</span>'
          + '<span class="pagemap-count">' + ev.section_data.length + ' sections</span>'
          + '<span class="pagemap-chev">\u203A</span>';
        hdr.addEventListener('click', () => card.classList.toggle('open'));
        card.appendChild(hdr);
        const sBody = document.createElement('div');
        sBody.className = 'pagemap-sections-body';
        const sWrap = document.createElement('div');
        sWrap.className = 'pagemap-sections';
        const VISIBLE = 5;
        const total = ev.section_data.length;
        ev.section_data.forEach((s, i) => {
          const sec = document.createElement('div');
          sec.className = 'pagemap-section' + (i >= VISIBLE ? ' hidden-section' : '');
          const sRow = document.createElement('div');
          sRow.className = 'pagemap-section-row';
          const rc = roleClass(s.role);
          sRow.innerHTML = '<span class="pagemap-section-dot ' + rc + '"></span>'
            + '<span class="pagemap-section-id" title="' + esc(s.id) + '">' + esc(s.id) + '</span>'
            + '<span style="flex:1"></span>'
            + (s.interactive ? '<span class="pagemap-section-badge">'
              + s.interactive + ' interactive</span>' : '')
            + '<span class="pagemap-section-chev">\u203A</span>';
          sec.appendChild(sRow);
          if (s.content) {
            const body = document.createElement('div');
            body.className = 'pagemap-section-content';
            const pre = document.createElement('pre');
            pre.className = 'pagemap-section-pre';
            pre.textContent = s.content;
            body.appendChild(pre);
            sec.appendChild(body);
          }
          sRow.addEventListener('click', (e) => { e.stopPropagation(); sec.classList.toggle('open'); });
          sWrap.appendChild(sec);
        });
        sBody.appendChild(sWrap);
        if (total > VISIBLE) {
          const more = document.createElement('button');
          more.className = 'pagemap-more';
          more.textContent = 'Show all ' + total + ' sections';
          more.addEventListener('click', (e) => { e.stopPropagation(); card.classList.add('show-all'); });
          sBody.appendChild(more);
        }
        card.appendChild(sBody);
        d.appendChild(card);
        feedAppend(d); scrollDown(); return;
      }

      if (t === 'zoom_view') {
        const d = document.createElement('div');
        d.className = 'zoomview';
        const card = document.createElement('div');
        card.className = 'zoomview-card';
        const hdr = document.createElement('div');
        hdr.className = 'zoomview-header';
        hdr.innerHTML = '<span class="zoomview-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff9f0a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="11" y1="8" x2="11" y2="14"/></svg></span>'
          + '<span style="flex-shrink:0">Zoomed section</span>'
          + '<span class="zoomview-ids">' + esc(ev.ids || '') + '</span>'
          + '<span class="zoomview-chev">\u203A</span>';
        hdr.addEventListener('click', () => card.classList.toggle('open'));
        card.appendChild(hdr);
        const body = document.createElement('div');
        body.className = 'zoomview-body';
        const pre = document.createElement('pre');
        pre.className = 'zoomview-pre';
        pre.innerHTML = hiHtml(ev.html || '');
        body.appendChild(pre);
        card.appendChild(body);
        d.appendChild(card);
        feedAppend(d); scrollDown(); return;
      }

      if (t === 'script_found') {
        const d = document.createElement('div');
        d.className = 'sys';
        d.innerHTML = '<span class="sys-dot"></span>'
          + (ev.valid ? 'Script extracted' : 'Script issue \u2014 ' + esc(ev.error || ''));
        feedAppend(d); scrollDown(); return;
      }

      if (t === 'script_running') {
        // ── Section divider ──
        const divider = document.createElement('div');
        divider.className = 'final-divider';
        divider.innerHTML = '<span class="final-divider-line"></span>'
          + '<span class="final-divider-label">Final Script</span>'
          + '<span class="final-divider-line"></span>';
        feedAppend(divider);

        const entry = document.createElement('div');
        entry.className = 'action final-script';
        const row = document.createElement('div');
        row.className = 'action-row';
        const ind = document.createElement('div');
        ind.className = 'action-ind';
        ind.innerHTML = _pySvg;
        const lbl = document.createElement('span');
        lbl.className = 'action-label';
        lbl.textContent = 'Running final script';
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
        feedAppend(entry); lastAction = entry; scrollDown();
        startActionTimer(meta, ev.timeout_budget || '');
        return;
      }

      if (t === 'script_output') {
        stopActionTimer();
        const target = lastAction;
        if (target) {
          const ok = ev.returncode === 0;
          const meta = target.querySelector('.action-meta');
          const sym = ok ? '\u2713' : '\u2717';
          const cls = ok ? 'action-meta-ok' : 'action-meta-err';
          let metaHtml = '<span class="' + cls + '">' + sym + '</span> ';
          if (ev.duration_s) {
            metaHtml += ev.duration_s + 's';
            if (ev.timeout_s) metaHtml += ' / ' + ev.timeout_s + 's';
          }
          meta.innerHTML = metaHtml;
          const detail = target.querySelector('.action-detail');
          if (ev.output) {
            detail.appendChild(makeActionOut(ev.output, !ok));
          }
          if (ev.sandbox_blocked) {
            const hint = document.createElement('div');
            hint.className = 'sandbox-hint';
            hint.innerHTML = '<span class="sandbox-hint-icon">\u26a0</span>'
              + '<span>Code blocked by sandbox \u2014 '
              + 'the AI used a restricted operation. '
              + 'It will retry with a safe alternative.</span>';
            detail.appendChild(hint);
          }
          // Auto-open the final script card so output is always visible.
          target.classList.add('open');
          lastAction = null;
        }
        scrollDown(); return;
      }

      if (t === 'approved') {
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card success">'
          + '<div class="terminal-row"><div class="terminal-icon">\u2713</div>'
          + '<div class="terminal-title">Script generated successfully</div></div></div>';
        feedAppend(d); setStatus('Complete', '#30d158'); scrollDown();
        return;
      }

      if (t === 'rejected') {
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card fail">'
          + '<div class="terminal-row"><div class="terminal-icon">\u2717</div>'
          + '<div class="terminal-title">Needs revision</div></div>'
          + (ev.feedback ? '<div class="terminal-sub">' + md(ev.feedback) + '</div>' : '')
          + '</div>';
        feedAppend(d); scrollDown(); return;
      }

      if (t === 'done') {
        const ok = ev.success;
        /* On success the 'approved' event already showed the final
           card and set the status — skip the duplicate. */
        if (ok) { return; }
        const d = document.createElement('div');
        d.className = 'terminal';
        d.innerHTML = '<div class="terminal-card fail">'
          + '<div class="terminal-icon">\u2717</div>'
          + '<div class="terminal-title">Failed</div>'
          + (ev.error ? '<div class="terminal-sub">' + md(ev.error) + '</div>' : '')
          + '</div>';
        feedAppend(d);
        setStatus('Failed', '#ff453a');
        scrollDown(); return;
      }

      if (t === 'results') {
        buildResultsViewer(ev.data);
        return;
      }

      if (t === 'system') {
        const d = document.createElement('div');
        d.className = 'sys';
        d.innerHTML = '<span class="sys-dot"></span>' + esc(ev.message || '');
        feedAppend(d); scrollDown(); return;
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
        feedAppend(d); scrollDown(); return;
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
        feedAppend(d);
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
        d.className = 'plan-loading';
        const { el: gen, meta: pct } = buildStatusEl('Generating plan');
        pct.textContent = '0%';
        d.appendChild(gen);
        feedAppend(d); scrollDown();
        const startTime = Date.now();
        const cap = 90;
        const k = 0.12;
        window.__planProgressTimer = setInterval(() => {
          const elapsed = (Date.now() - startTime) / 1000;
          const progress = Math.min(cap, cap * (1 - Math.exp(-k * elapsed)));
          pct.textContent = Math.round(progress) + '%';
        }, 200);
        return;
      }

      if (t === 'plan') {
        if (window.__planProgressTimer) { clearInterval(window.__planProgressTimer); window.__planProgressTimer = null; }
        const loader = feed.querySelector('.plan-loading');
        if (loader) {
          const pctLabel = loader.querySelector('.agent-status-meta');
          if (pctLabel) pctLabel.textContent = '100%';
        }
        const showPlan = () => {
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
            + '<span class="plan-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></span>'
            + '<span class="plan-title">Exploration plan</span>'
            + '<span class="plan-chevron">\u203A</span>'
            + '</div><div class="plan-body"><div class="plan-content">'
            + listHtml
            + '</div></div></div>';
          d.innerHTML = html;
          d.querySelector('.plan-header').addEventListener('click', () => d.classList.toggle('open'));
          if (loader) {
            loader.replaceWith(d);
          } else {
            feedAppend(d);
          }
          scrollDown();
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
        self._main_page = None
        self._hl_ready = False
        self._injected = False
        self._events: list[dict[str, Any]] = []

    async def init(self, overlay_page, *, main_page=None) -> None:
        """Load the overlay UI into the popup page.

        Args:
            overlay_page: The popup page for the event panel.
            main_page: The website page (for section highlighting).
        """
        self._overlay_page = overlay_page
        self._main_page = main_page
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

        # Register the expand callback early — expose_function triggers
        # Patchright's install_inject_route() which can cause Chrome to
        # reposition popup windows.  Doing it here (once, at init) avoids
        # the side-effect when results are pushed later.
        try:
            await overlay_page.expose_function(
                "__scout_expand_cb",
                lambda expanded: asyncio.ensure_future(
                    self._resize_windows(bool(expanded)),
                ),
            )
            await overlay_page.evaluate(
                "window.__scout_expand = (x) => __scout_expand_cb(x)",
            )
        except Exception:
            logger.debug("[overlay] expose __scout_expand_cb failed",
                         exc_info=True)

        if main_page is not None:
            await self._setup_highlight_host()

    # Event types with large payloads that are safe to drop on
    # replay — they carry supplementary UI content, not critical
    # state.  Excluding them keeps the replay buffer lean and
    # prevents Playwright serialisation failures.
    _SKIP_BUFFER_TYPES = frozenset({
        "page_update", "zoom_view", "results",
        "thinking_start", "thinking_end",
    })

    async def push(self, event: dict[str, Any]) -> None:
        """Buffer the event and push it to the overlay."""
        if event.get("type") not in self._SKIP_BUFFER_TYPES:
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
            # Re-establish the JS→Python bridge for the expand callback.
            # expose_function persists on the Python side across recovery,
            # but set_content wipes the JS-side function reference.
            try:
                await self._overlay_page.evaluate(
                    "window.__scout_expand = (x) => __scout_expand_cb(x)",
                )
            except Exception:
                pass
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

    async def push_thinking_start(self) -> None:
        """Show the animated thinking indicator."""
        await self.push({"type": "thinking_start"})

    async def push_thinking_end(self) -> None:
        """Remove the animated thinking indicator."""
        await self.push({"type": "thinking_end"})

    async def push_thinking(self, text: str) -> None:
        await self.push({"type": "thinking", "text": text})

    async def push_tool_call(
        self,
        code: str,
        step: int = 0,
        max_steps: int = 0,
        *,
        label: str = "",
        timeout_budget: str = "",
    ) -> None:
        await self.push({
            "type": "tool_call",
            "code": code,
            "step": step,
            "max_steps": max_steps,
            "label": label,
            "timeout_budget": timeout_budget,
        })

    async def push_tool_result(
        self,
        *,
        is_error: bool = False,
        duration_s: str = "",
        output: str = "",
        error: str = "",
        timeout_info: str = "",
    ) -> None:
        await self.push({
            "type": "tool_result",
            "is_error": is_error,
            "duration_s": duration_s,
            "output": output,
            "error": error,
            "timeout_info": timeout_info,
        })

    async def push_page_update(
        self,
        url: str,
        sections: int,
        section_data: list[dict] | None = None,
    ) -> None:
        await self.push({
            "type": "page_update",
            "url": url,
            "sections": sections,
            "section_data": section_data or [],
        })

    async def push_zoom_view(self, ids: str, html: str) -> None:
        await self.push({
            "type": "zoom_view",
            "ids": ids,
            "html": html,
        })

    # ── Full-viewport "show_page" overlay ────────────────────────

    async def show_page_overlay(
        self, message: str = "Looking at the page\u2026",
    ) -> None:
        """Draw a full-viewport overlay with marching-ants border while the AI reads."""
        if not self._main_page:
            return
        if not self._hl_ready:
            await self._setup_highlight_host()
        try:
            await self._main_page.evaluate(
                r"""(msg) => {
                    const host = document.getElementById('scout-hl-host');
                    if (!host || !host.shadowRoot) return;
                    const shadow = host.shadowRoot;

                    /* Ensure .hl-zone-page exists. */
                    let zone = shadow.querySelector('.hl-zone-page');
                    if (!zone) {
                        zone = document.createElement('div');
                        zone.className = 'hl-zone-page';
                        shadow.appendChild(zone);
                    }
                    zone.innerHTML = '';

                    const style = document.createElement('style');
                    style.textContent = `
                        /* Marching-ants border — same pattern as zoom overlays */
                        @keyframes scout-page-march {
                            0%   { background-position: 0 0, 0 0, 100% 0, 0 100%; }
                            100% { background-position: 0 -24px, 24px 0, 100% 24px, -24px 100%; }
                        }
                        /* Background blinks between two alpha levels */
                        @keyframes scout-page-bg-pulse {
                            0%, 100% { background: rgba(15, 23, 42, 0.18); }
                            50%      { background: rgba(15, 23, 42, 0.08); }
                        }
                        /* Pill entrance */
                        @keyframes scout-page-pill-in {
                            from { opacity: 0; transform: translateY(8px) scale(0.96); }
                            to   { opacity: 1; transform: translateY(0) scale(1); }
                        }
                        /* Dot breathing */
                        @keyframes scout-page-dot-breathe {
                            0%, 100% { opacity: 1; transform: scale(1); }
                            50%      { opacity: 0.5; transform: scale(0.8); }
                        }

                        .hl-page-overlay {
                            position: fixed;
                            top: 0; left: 0;
                            width: 100vw; height: 100vh;
                            pointer-events: none;
                            z-index: 2147483647;
                        }

                        /* Pulsing semi-transparent background fill */
                        .hl-page-bg {
                            position: absolute;
                            inset: 0;
                            background: rgba(15, 23, 42, 0.18);
                            animation: scout-page-bg-pulse 2.8s ease-in-out infinite;
                        }

                        /* Marching-ants border around the entire viewport */
                        .hl-page-border {
                            position: absolute;
                            inset: 0;
                            border-radius: 0;
                            pointer-events: none;
                            background-color: transparent;
                            background-image:
                                linear-gradient(0deg,   rgba(59,130,246,0.55) 50%, transparent 50%),
                                linear-gradient(90deg,  rgba(59,130,246,0.55) 50%, transparent 50%),
                                linear-gradient(0deg,   rgba(59,130,246,0.55) 50%, transparent 50%),
                                linear-gradient(90deg,  rgba(59,130,246,0.55) 50%, transparent 50%);
                            background-size: 2px 12px, 12px 2px, 2px 12px, 12px 2px;
                            background-repeat: repeat-y, repeat-x, repeat-y, repeat-x;
                            background-position: 0 0, 0 0, 100% 0, 0 100%;
                            animation: scout-page-march 0.5s linear infinite;
                        }

                        /* Solid pill label — not transparent */
                        .hl-page-pill {
                            position: absolute;
                            top: 50%; left: 50%;
                            transform: translate(-50%, -50%);
                            display: inline-flex;
                            align-items: center;
                            gap: 10px;
                            padding: 10px 24px;
                            border-radius: 28px;
                            background: rgba(15, 23, 42, 0.88);
                            backdrop-filter: blur(12px);
                            -webkit-backdrop-filter: blur(12px);
                            border: 1px solid rgba(59, 130, 246, 0.35);
                            box-shadow: 0 4px 24px rgba(0,0,0,0.4),
                                        0 0 0 1px rgba(59,130,246,0.15);
                            animation: scout-page-pill-in 0.35s ease-out;
                        }
                        .hl-page-dot {
                            width: 8px; height: 8px;
                            border-radius: 50%;
                            background: rgb(59, 130, 246);
                            animation: scout-page-dot-breathe 2.8s ease-in-out infinite;
                        }
                        .hl-page-label {
                            font: 500 14px/1 -apple-system, BlinkMacSystemFont,
                                  "Segoe UI", Roboto, sans-serif;
                            color: rgba(255, 255, 255, 0.92);
                            letter-spacing: 0.01em;
                            white-space: nowrap;
                        }
                    `;
                    zone.appendChild(style);

                    const overlay = document.createElement('div');
                    overlay.className = 'hl-page-overlay';

                    /* Semi-transparent pulsing background */
                    const bg = document.createElement('div');
                    bg.className = 'hl-page-bg';
                    overlay.appendChild(bg);

                    /* Marching-ants border */
                    const border = document.createElement('div');
                    border.className = 'hl-page-border';
                    overlay.appendChild(border);

                    /* Solid pill with label */
                    const pill = document.createElement('div');
                    pill.className = 'hl-page-pill';

                    const dot = document.createElement('div');
                    dot.className = 'hl-page-dot';

                    const label = document.createElement('div');
                    label.className = 'hl-page-label';
                    label.textContent = msg;

                    pill.appendChild(dot);
                    pill.appendChild(label);
                    overlay.appendChild(pill);
                    zone.appendChild(overlay);
                }""",
                message,
            )
        except Exception:
            logger.debug("show_page overlay failed", exc_info=True)

    async def hide_page_overlay(self) -> None:
        """Remove the full-viewport show_page overlay."""
        if not self._main_page:
            return
        try:
            await self._main_page.evaluate(
                """() => {
                    const host = document.getElementById('scout-hl-host');
                    if (!host || !host.shadowRoot) return;
                    const zone = host.shadowRoot.querySelector('.hl-zone-page');
                    if (zone) zone.innerHTML = '';
                }""",
            )
        except Exception:
            pass

    async def push_script_found(
        self, valid: bool, error: str = "",
    ) -> None:
        await self.push({
            "type": "script_found",
            "valid": valid,
            "error": error,
        })

    async def push_script_running(
        self, *, timeout_budget: str = "",
    ) -> None:
        await self.push({
            "type": "script_running",
            "timeout_budget": timeout_budget,
        })

    async def push_script_output(
        self, output: str, returncode: int,
        *, duration_s: str = "", timeout_s: str = "",
        sandbox_blocked: bool = False,
    ) -> None:
        await self.push({
            "type": "script_output",
            "output": output,
            "returncode": returncode,
            "duration_s": duration_s,
            "timeout_s": timeout_s,
            "sandbox_blocked": sandbox_blocked,
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

    # ── JSON Results Viewer ───────────────────────────────────────

    async def push_results(self, json_data: str) -> None:
        """Show the interactive JSON results viewer in the overlay.

        Must be called after ``push_done()`` while the browser is still
        open.  The ``__scout_expand`` callback is already registered
        during ``inject()`` so there are no page-level side effects here.

        Unlike normal events, ``results`` is pushed directly without the
        generic ``push()`` path.  The payload can be very large, and a
        failed ``evaluate`` must **not** trigger ``_recover()`` — which
        does ``set_content()`` and can cause Chrome to reposition the
        popup window.
        """
        if not self._overlay_page or not self._injected:
            return

        # Push directly — do NOT use self.push() which triggers _recover
        # on failure.  Results are also excluded from the replay buffer.
        try:
            await self._overlay_page.evaluate(
                "ev => window.__scout.pushEvent(ev)",
                {"type": "results", "data": json_data},
            )
        except Exception:
            logger.debug("[overlay] results push failed", exc_info=True)

    async def _resize_windows(self, expanded: bool) -> None:
        """Resize overlay and main windows for expand/collapse toggle."""
        if not self._main_page or not self._overlay_page:
            return
        from ..browser import compute_demo_layout, compute_expanded_layout

        try:
            screen = await self._main_page.evaluate(
                "({ w: screen.availWidth, h: screen.availHeight })",
            )
            sw, sh = screen["w"], screen["h"]
        except Exception:
            sw, sh = 1920, 1080

        layout_fn = compute_expanded_layout if expanded else compute_demo_layout
        layout = layout_fn(sw, sh)
        pw = layout["page_width"]
        ph = layout["height"]
        panelw = layout["panel_width"]
        panelx = layout["panel_x"]

        # Use CDP Browser.setWindowBounds for reliable positioning —
        # window.moveTo() is unreliable for popup windows in Chrome.
        async def _cdp_set_bounds(page, x, y, w, h, label=""):
            try:
                cdp = await page.context.new_cdp_session(page)
                target = await cdp.send("Browser.getWindowForTarget")
                wid = target["windowId"]
                await cdp.send("Browser.setWindowBounds", {
                    "windowId": wid,
                    "bounds": {
                        "left": x, "top": y,
                        "width": w, "height": h,
                        "windowState": "normal",
                    },
                })
                await cdp.detach()
                logger.info(
                    "[overlay] CDP setWindowBounds %s: x=%d w=%d",
                    label, x, w,
                )
            except Exception:
                logger.info(
                    "[overlay] CDP setWindowBounds %s failed, "
                    "falling back to JS",
                    label, exc_info=True,
                )
                # Fallback to JS resize.
                try:
                    await page.evaluate(
                        f"window.moveTo({x},{y});"
                        f"window.resizeTo({w},{h})",
                    )
                except Exception:
                    pass

        logger.info(
            "[overlay] _resize_windows expanded=%s → "
            "main(0,0,%d,%d) overlay(%d,0,%d,%d)",
            expanded, pw, ph, panelx, panelw, ph,
        )
        await _cdp_set_bounds(self._main_page, 0, 0, pw, ph, "main")
        await _cdp_set_bounds(self._overlay_page, panelx, 0, panelw, ph, "overlay")

    async def wait_for_dismiss(
        self,
        *,
        main_page: Any = None,
        timeout_s: float = 600.0,
    ) -> None:
        """Block until the user clicks Finish, closes browser, or timeout.

        Racing conditions:
          1. User clicks "Finish" → JS calls __scout_dismiss_cb → resolves
          2. User closes overlay (X) → pill injected on website, keep waiting
          3. User closes browser window → resolves immediately
          4. Timeout (default 10 min) → resolves with warning
        """
        if not self._overlay_page:
            return

        loop = asyncio.get_event_loop()
        dismiss_future: asyncio.Future[str] = loop.create_future()

        def _resolve(reason: str) -> None:
            if not dismiss_future.done():
                logger.info("[overlay] dismiss resolved: %s", reason)
                dismiss_future.set_result(reason)

        # Use page.evaluate with a JS Promise that resolves when the
        # user clicks Finish.  This avoids expose_function race conditions.
        # We set window.__scout_dismiss as a resolver; the Finish button
        # (created by buildResultsViewer) calls it on click.

        # Handler: overlay page closed → inject pill, keep waiting.
        def _on_overlay_close() -> None:
            logger.info("[overlay] overlay page close event fired")
            if dismiss_future.done():
                return
            asyncio.ensure_future(
                self._inject_floating_pill(main_page, dismiss_future),
            )

        # Handler: main browser page closed → finish immediately.
        def _on_main_close() -> None:
            logger.info("[overlay] main page close event fired")
            _resolve("browser_closed")

        overlay_ref = self._overlay_page
        overlay_ref.on("close", _on_overlay_close)
        if main_page:
            main_page.on("close", _on_main_close)

        # Start a background task that awaits the JS Promise.
        # The Promise resolves when window.__scout_dismiss() is called.
        # If the user clicked Finish in the gap before this runs,
        # __scout_dismiss_pending is set — resolve immediately.
        async def _await_js_dismiss() -> None:
            try:
                await self._overlay_page.evaluate(
                    "new Promise(resolve => {"
                    "  window.__scout_dismiss = resolve;"
                    "})",
                )
                _resolve("finish_clicked")
            except Exception:
                # Page closed or crashed — the close handlers will deal with it.
                logger.debug("[overlay] JS dismiss promise rejected",
                             exc_info=True)

        dismiss_task = asyncio.ensure_future(_await_js_dismiss())

        try:
            await asyncio.wait_for(dismiss_future, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.info(
                "[overlay] results viewer timed out after %.0fs", timeout_s,
            )
        finally:
            dismiss_task.cancel()
            try:
                overlay_ref.remove_listener("close", _on_overlay_close)
            except Exception:
                pass
            if main_page:
                try:
                    main_page.remove_listener("close", _on_main_close)
                except Exception:
                    pass

    async def _inject_floating_pill(
        self,
        main_page: Any,
        dismiss_future: asyncio.Future[str],
    ) -> None:
        """Inject a floating pill into the website when the overlay is closed.

        Provides a "Finish" button so the user can still signal completion
        without the overlay panel.
        """
        if not main_page or main_page.is_closed():
            if not dismiss_future.done():
                dismiss_future.set_result("page_gone")
            return

        def _pill_dismiss() -> None:
            if not dismiss_future.done():
                dismiss_future.set_result("pill_finish")

        try:
            await main_page.expose_function(
                "__scout_pill_dismiss", _pill_dismiss,
            )
        except Exception:
            # May fail if already registered from a previous overlay close.
            pass

        pill_js = r"""(() => {
            if (document.getElementById('scout-pill')) return;
            const style = document.createElement('style');
            style.textContent = `
                @keyframes scout-pill-in {
                    from { opacity: 0; transform: translateY(10px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                #scout-pill {
                    position: fixed; bottom: 20px; right: 20px;
                    z-index: 2147483647;
                    display: flex; align-items: center; gap: 10px;
                    padding: 10px 12px 10px 16px; border-radius: 24px;
                    background: rgba(22, 22, 24, 0.92);
                    backdrop-filter: blur(20px) saturate(180%);
                    -webkit-backdrop-filter: blur(20px) saturate(180%);
                    border: 1px solid rgba(255,255,255,0.1);
                    box-shadow: 0 4px 24px rgba(0,0,0,0.4),
                                0 0 0 0.5px rgba(255,255,255,0.05);
                    font: 500 13px/1 -apple-system, BlinkMacSystemFont,
                          "SF Pro Text", "Helvetica Neue", sans-serif;
                    color: rgba(255,255,255,0.85);
                    animation: scout-pill-in 0.3s cubic-bezier(0.4,0,0.2,1);
                }
                #scout-pill .pill-label {
                    color: rgba(255,255,255,0.35);
                    font-weight: 600; letter-spacing: 0.02em;
                }
                #scout-pill .pill-sep {
                    color: rgba(255,255,255,0.1);
                    font-weight: 300;
                }
                #scout-pill .pill-finish {
                    background: linear-gradient(135deg, #5e5ce6, #bf5af2);
                    color: #fff; font-weight: 600; font-size: 12px;
                    border: none; cursor: pointer;
                    padding: 6px 14px; border-radius: 8px;
                    transition: filter 0.15s ease, transform 0.1s ease;
                }
                #scout-pill .pill-finish:hover {
                    filter: brightness(1.12);
                }
                #scout-pill .pill-finish:active {
                    transform: scale(0.96); filter: brightness(0.95);
                }
            `;
            document.head.appendChild(style);

            const pill = document.createElement('div');
            pill.id = 'scout-pill';

            const label = document.createElement('span');
            label.className = 'pill-label';
            label.textContent = 'Scout';

            const sep = document.createElement('span');
            sep.className = 'pill-sep';
            sep.textContent = '|';

            const btn = document.createElement('button');
            btn.className = 'pill-finish';
            btn.textContent = 'Finish';
            btn.addEventListener('click', () => {
                window.__scout_pill_dismiss();
                pill.style.transition = 'opacity 0.2s, transform 0.2s';
                pill.style.opacity = '0';
                pill.style.transform = 'translateY(10px)';
                setTimeout(() => pill.remove(), 250);
            });

            pill.append(label, sep, btn);
            document.body.appendChild(pill);
        })()"""

        try:
            await main_page.evaluate(pill_js)
        except Exception:
            if not dismiss_future.done():
                dismiss_future.set_result("pill_inject_failed")

    # ── Section highlighting (main page) ───────────────────────────

    async def _setup_highlight_host(self) -> None:
        """Inject the shadow DOM container for highlights into the main page."""
        try:
            await self._main_page.evaluate(
                """() => {
                    if (document.getElementById('scout-hl-host')) return;
                    const host = document.createElement('div');
                    host.id = 'scout-hl-host';
                    host.style.cssText =
                        'position:fixed;top:0;left:0;width:0;height:0;'
                        + 'pointer-events:none;z-index:2147483647';
                    document.documentElement.appendChild(host);
                    const shadow = host.attachShadow({ mode: 'open' });
                    shadow.innerHTML =
                        '<div class="hl-zone-zoom"></div>'
                        + '<div class="hl-zone-interact"></div>';
                }""",
            )
            self._hl_ready = True
        except Exception:
            logger.debug("Highlight host setup failed", exc_info=True)
            self._hl_ready = False

    async def highlight_sections(self, selectors: list[str]) -> None:
        """Draw animated marching-ant overlays on the main page.

        Each CSS selector is resolved to an element; a fixed-position
        overlay div is placed over its bounding rect inside a shadow
        DOM container.  A ``requestAnimationFrame`` loop keeps the
        overlays pinned to the target elements through scroll/resize.
        """
        if not self._main_page or not selectors:
            return
        try:
            await self._main_page.evaluate(
                r"""(selectors) => {
                    let host = document.getElementById('scout-hl-host');
                    if (!host) {
                        host = document.createElement('div');
                        host.id = 'scout-hl-host';
                        host.style.cssText =
                            'position:fixed;top:0;left:0;width:0;height:0;'
                            + 'pointer-events:none;z-index:2147483647';
                        document.documentElement.appendChild(host);
                        const s = host.attachShadow({ mode: 'open' });
                        s.innerHTML =
                            '<div class="hl-zone-zoom"></div>'
                            + '<div class="hl-zone-interact"></div>';
                    }
                    const shadow = host.shadowRoot;
                    let zone = shadow.querySelector('.hl-zone-zoom');
                    if (!zone) {
                        shadow.innerHTML =
                            '<div class="hl-zone-zoom"></div>'
                            + '<div class="hl-zone-interact"></div>';
                        zone = shadow.querySelector('.hl-zone-zoom');
                    }
                    zone.innerHTML = '';

                    /* Ensure unified state exists */
                    if (!host.__scoutState) {
                        host.__scoutState = {
                            rafId: null,
                            running: false,
                            interactTracked: [],
                            zoomTracked: [],
                        };
                    }
                    const state = host.__scoutState;

                    const style = document.createElement('style');
                    style.textContent = `
                        @keyframes scout-march {
                            0%   { background-position: 0 0, 0 0, 100% 0, 0 100%; }
                            100% { background-position: 0 -24px, 24px 0, 100% 24px, -24px 100%; }
                        }
                        @keyframes scout-fadein {
                            from { opacity: 0; }
                            to   { opacity: 1; }
                        }
                        @keyframes scout-cursor {
                            from { border-right-color: rgba(255,255,255,0.8); }
                            to   { border-right-color: transparent; }
                        }
                        .hl {
                            position: fixed;
                            pointer-events: none;
                            border-radius: 6px;
                            background-color: rgba(37,99,235,0.14);
                            background-image:
                                linear-gradient(0deg,   rgba(59,130,246,0.45) 50%, transparent 50%),
                                linear-gradient(90deg,  rgba(59,130,246,0.45) 50%, transparent 50%),
                                linear-gradient(0deg,   rgba(59,130,246,0.45) 50%, transparent 50%),
                                linear-gradient(90deg,  rgba(59,130,246,0.45) 50%, transparent 50%);
                            background-size: 2px 12px, 12px 2px, 2px 12px, 12px 2px;
                            background-repeat: repeat-y, repeat-x, repeat-y, repeat-x;
                            box-shadow: inset 0 0 30px rgba(37,99,235,0.06),
                                        0 0 12px rgba(59,130,246,0.10);
                            animation:
                                scout-march  0.4s linear infinite,
                                scout-fadein 0.35s ease;
                        }
                        .hl-label {
                            position: absolute;
                            top: -1px;
                            left: 8px;
                            transform: translateY(-100%);
                            padding: 3px 10px;
                            border-radius: 4px 4px 0 0;
                            background: rgba(30,64,175,0.85);
                            font: 500 11px/1.2 -apple-system, BlinkMacSystemFont,
                                  "Segoe UI", Roboto, sans-serif;
                            color: rgba(255,255,255,0.92);
                            letter-spacing: 0.3px;
                            white-space: nowrap;
                            overflow: hidden;
                            border-right: 2px solid rgba(255,255,255,0.8);
                            animation: scout-cursor 0.55s step-end infinite;
                        }
                    `;
                    zone.appendChild(style);

                    const labelText = 'Inspecting HTML\u2026';
                    const tracked = [];

                    for (const sel of selectors) {
                        try {
                            const el = document.querySelector(sel);
                            if (!el) continue;
                            const rect = el.getBoundingClientRect();
                            if (rect.width === 0 && rect.height === 0) continue;
                            const d = document.createElement('div');
                            d.className = 'hl';
                            d.style.top    = rect.top  + 'px';
                            d.style.left   = rect.left + 'px';
                            d.style.width  = rect.width  + 'px';
                            d.style.height = rect.height + 'px';

                            const lbl = document.createElement('span');
                            lbl.className = 'hl-label';
                            d.appendChild(lbl);
                            zone.appendChild(d);

                            /* Track for rAF position updates */
                            tracked.push({ selector: sel, div: d });

                            /* Typewriter effect */
                            let ci = 0;
                            const typeChar = () => {
                                if (ci < labelText.length) {
                                    lbl.textContent += labelText[ci++];
                                    setTimeout(typeChar, 35 + Math.random() * 25);
                                }
                            };
                            typeChar();
                        } catch(e) { /* skip silently */ }
                    }

                    /* Merge into unified state and ensure rAF loop runs */
                    state.zoomTracked = tracked;

                    if (!state.running) {
                        function updatePositions() {
                            const h = document.getElementById('scout-hl-host');
                            if (!h || !h.__scoutState || !h.__scoutState.running) return;
                            const st = h.__scoutState;
                            for (const arr of [st.interactTracked, st.zoomTracked]) {
                                for (const entry of arr) {
                                    try {
                                        const el = document.querySelector(entry.selector);
                                        if (!el) { entry.div.style.display = 'none'; continue; }
                                        const r = el.getBoundingClientRect();
                                        if (r.width === 0 && r.height === 0) {
                                            entry.div.style.display = 'none'; continue;
                                        }
                                        entry.div.style.display = '';
                                        entry.div.style.top    = r.top + 'px';
                                        entry.div.style.left   = r.left + 'px';
                                        entry.div.style.width  = r.width + 'px';
                                        entry.div.style.height = r.height + 'px';
                                    } catch(e) { entry.div.style.display = 'none'; }
                                }
                            }
                            st.rafId = requestAnimationFrame(updatePositions);
                        }
                        state.running = true;
                        state.rafId = requestAnimationFrame(updatePositions);
                    }
                }""",
                selectors,
            )
            self._hl_ready = True
        except Exception:
            logger.debug("Highlight draw failed", exc_info=True)

    async def clear_highlights(self) -> None:
        """Remove all highlight overlays and stop the rAF tracking loop."""
        if not self._hl_ready or not self._main_page:
            return
        try:
            await self._main_page.evaluate(
                """() => {
                    const host = document.getElementById('scout-hl-host');
                    if (!host || !host.shadowRoot) return;

                    /* Stop rAF loop and clear tracking state */
                    const st = host.__scoutState;
                    if (st) {
                        if (st.rafId) cancelAnimationFrame(st.rafId);
                        st.running = false;
                        st.interactTracked = [];
                        st.zoomTracked = [];
                        host.__scoutState = null;
                    }

                    const shadow = host.shadowRoot;
                    const zoom = shadow.querySelector('.hl-zone-zoom');
                    const interact = shadow.querySelector('.hl-zone-interact');
                    const page = shadow.querySelector('.hl-zone-page');
                    if (zoom) zoom.innerHTML = '';
                    if (interact) interact.innerHTML = '';
                    if (page) page.innerHTML = '';
                }""",
            )
        except Exception:
            pass

    # ── Interaction highlighting (main page) ──────────────────────

    _MAX_QSA_HIGHLIGHTS = None  # unlimited

    async def _resolve_css_path(
        self, selector: str,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Resolve a Playwright selector to ``(bounding_box, css_path)``.

        The CSS path is a unique selector string (using ``tagName``,
        ``id``, and ``:nth-child``) that browser JS can use in
        ``document.querySelector()`` for continuous rAF tracking.

        Falls back gracefully: if the CSS path cannot be computed the
        bounding box is still returned (overlay will be static).
        """
        try:
            elements = await self._main_page.query_selector_all(selector)
        except Exception:
            return None, None
        if not elements:
            return None, None

        vp = self._main_page.viewport_size
        vp_w = vp["width"] if vp else 1920
        vp_h = vp["height"] if vp else 1080

        fallback: tuple[Any, dict[str, float]] | None = None
        for el in elements:
            try:
                box = await el.bounding_box()
            except Exception:
                continue
            if not box or (box["width"] == 0 and box["height"] == 0):
                continue
            if fallback is None:
                fallback = (el, box)
            # Prefer element that overlaps the viewport
            if (box["y"] + box["height"] > 0
                    and box["y"] < vp_h
                    and box["x"] + box["width"] > 0
                    and box["x"] < vp_w):
                try:
                    css_path = await el.evaluate(_CSS_PATH_JS)
                    return box, css_path
                except Exception:
                    return box, None

        if fallback:
            el, box = fallback
            try:
                css_path = await el.evaluate(_CSS_PATH_JS)
                return box, css_path
            except Exception:
                return box, None
        return None, None

    async def highlight_interactions(
        self, results: list[ExtractionResult],
        *,
        _deferred: bool = False,
    ) -> dict[str, Any]:
        """Draw interaction overlays on elements the AI code targets.

        Consumes :class:`ExtractionResult` objects from the Selector
        Extractor.  Each result is resolved to a bounding box and
        rendered with a visual style based on its ``action_category``:

        - **navigating** — orange/amber pulse, self-removing (~600ms)
        - **mutating** — purple/blue soft glow, persistent
        - **passive** — teal subtle outline, persistent

        Results with ``in_loop=True`` or ``after_navigation=True`` are
        skipped.  All failures are caught silently.

        Returns:
            A dict with observability stats for console logging.
        """
        empty_stats: dict[str, Any] = {
            "resolved_count": 0,
            "drawn_count": 0,
            "dropped_overlap": 0,
            "details": [],
            "deferred": [],
        }

        if not self._main_page or not results:
            return empty_stats

        # ── Filter ───────────────────────────────────────────────
        # in_loop filter disabled: rAF tracking only sees the final
        # state after the loop completes, so loop selectors are safe.
        # To re-enable in_loop filtering, restore: not r.in_loop and
        #
        # after_navigation results are deferred — they're returned in
        # the stats dict so the caller can draw them AFTER code
        # execution, once the new page has loaded.
        # When called with _deferred=True, skip the nav filter (the
        # caller already waited for navigation to complete).
        if _deferred:
            filtered = list(results)
            deferred = []
        else:
            filtered = [
                r for r in results
                if not r.after_navigation
            ]
            deferred = [
                r for r in results
                if r.after_navigation
            ]

        # Track per-selector status for logging
        details: list[dict[str, str]] = []
        for r in deferred:
            details.append({
                "selector": r.selector,
                "category": r.action_category,
                "action": r.action,
                "source": r.source,
                "status": "deferred",
                "reason": "after_nav",
            })

        if not filtered:
            return {**empty_stats, "details": details, "deferred": deferred}

        # ── Resolve selectors to bounding-box data ──────────────
        payload: list[dict[str, Any]] = []
        idx = 0
        total_items = len(filtered)

        for r in filtered:
            try:
                if r.selector_type == "playwright":
                    if r.action == "query_selector_all":
                        elements = await self._main_page.query_selector_all(
                            r.selector,
                        )
                        total_count = len(elements)
                        if not elements:
                            details.append({
                                "selector": r.selector,
                                "category": r.action_category,
                                "action": r.action,
                                "source": r.source,
                                "status": "not_found",
                            })
                            continue
                        for el in elements:
                            try:
                                box = await el.bounding_box()
                            except Exception:
                                continue
                            if box and box["width"] > 0 and box["height"] > 0:
                                # Compute CSS path for rAF tracking
                                css_path = None
                                try:
                                    css_path = await el.evaluate(_CSS_PATH_JS)
                                except Exception:
                                    pass
                                item: dict[str, Any] = {
                                    "rect": box,
                                    "category": r.action_category,
                                    "action": r.action,
                                    "totalCount": total_count,
                                    "index": idx,
                                    "totalItems": total_items,
                                }
                                if css_path:
                                    item["trackSelector"] = css_path
                                payload.append(item)
                                idx += 1
                        details.append({
                            "selector": r.selector,
                            "category": r.action_category,
                            "action": r.action,
                            "source": r.source,
                            "status": "resolved",
                            "count": str(total_count),
                        })
                    else:
                        # Resolve to bounding box + CSS path for rAF
                        box, css_path = await self._resolve_css_path(
                            r.selector,
                        )
                        if box:
                            item = {
                                "rect": box,
                                "category": r.action_category,
                                "action": r.action,
                                "totalCount": 0,
                                "index": idx,
                                "totalItems": total_items,
                            }
                            if css_path:
                                item["trackSelector"] = css_path
                            payload.append(item)
                            idx += 1
                            details.append({
                                "selector": r.selector,
                                "category": r.action_category,
                                "action": r.action,
                                "source": r.source,
                                "status": "resolved",
                            })
                        else:
                            details.append({
                                "selector": r.selector,
                                "category": r.action_category,
                                "action": r.action,
                                "source": r.source,
                                "status": "not_found",
                            })
                else:
                    # CSS selector — pass as trackSelector for rAF
                    payload.append({
                        "selector": r.selector,
                        "trackSelector": r.selector,
                        "category": r.action_category,
                        "action": r.action,
                        "totalCount": 0,
                        "index": idx,
                        "totalItems": total_items,
                    })
                    idx += 1
                    details.append({
                        "selector": r.selector,
                        "category": r.action_category,
                        "action": r.action,
                        "source": r.source,
                        "status": "resolved",
                        "note": "css (JS-side)",
                    })
            except Exception:
                details.append({
                    "selector": r.selector,
                    "category": r.action_category,
                    "action": r.action,
                    "source": r.source,
                    "status": "not_found",
                    "reason": "error",
                })
                continue  # graceful degradation

        if not payload:
            return {
                "resolved_count": 0,
                "drawn_count": 0,
                "dropped_overlap": 0,
                "details": details,
                "deferred": deferred,
            }

        # ── Single page.evaluate() to draw all overlays ─────────
        js_stats: dict[str, int] = {}
        try:
            js_stats = await self._main_page.evaluate(
                _INTERACTION_HIGHLIGHT_JS,
                payload,
            ) or {}
            self._hl_ready = True
        except Exception:
            logger.debug("Interaction highlight draw failed", exc_info=True)

        resolved_count = js_stats.get("resolved", len(payload))
        kept_count = js_stats.get("kept", len(payload))
        dropped = js_stats.get("dropped", 0)
        js_statuses = js_stats.get("statuses", [])

        # Update detail entries with JS-side status (drawn/overlap/not_found)
        if js_statuses:
            # Map payload index back to detail entries that were "resolved"
            payload_idx = 0
            for d in details:
                if d["status"] == "resolved" and payload_idx < len(js_statuses):
                    d["status"] = js_statuses[payload_idx]
                    payload_idx += 1

        return {
            "resolved_count": resolved_count,
            "drawn_count": kept_count,
            "dropped_overlap": dropped,
            "details": details,
            "deferred": deferred,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  CSS path computation — run via element.evaluate() in Python
# ═══════════════════════════════════════════════════════════════════════════

_CSS_PATH_JS = """(el) => {
    const parts = [];
    while (el && el !== document.body && el !== document.documentElement) {
        let sel = el.tagName.toLowerCase();
        if (el.id) {
            try {
                if (document.querySelectorAll('#' + CSS.escape(el.id)).length === 1) {
                    parts.unshift(sel + '#' + CSS.escape(el.id));
                    return parts.join(' > ');
                }
            } catch(e) {}
        }
        const parent = el.parentElement;
        if (parent) {
            const idx = Array.from(parent.children).indexOf(el) + 1;
            sel += ':nth-child(' + idx + ')';
        }
        parts.unshift(sel);
        el = el.parentElement;
    }
    return parts.join(' > ');
}"""


# ═══════════════════════════════════════════════════════════════════════════
#  Interaction highlight JS — injected via page.evaluate()
# ═══════════════════════════════════════════════════════════════════════════

_INTERACTION_HIGHLIGHT_JS = r"""(items) => {
    /* ── Find or create shadow host + zones ───────────────── */
    let host = document.getElementById('scout-hl-host');
    if (!host) {
        host = document.createElement('div');
        host.id = 'scout-hl-host';
        host.style.cssText =
            'position:fixed;top:0;left:0;width:0;height:0;'
            + 'pointer-events:none;z-index:2147483647';
        document.documentElement.appendChild(host);
        const s = host.attachShadow({ mode: 'open' });
        s.innerHTML =
            '<div class="hl-zone-zoom"></div>'
            + '<div class="hl-zone-interact"></div>';
    }
    const shadow = host.shadowRoot;
    let zone = shadow.querySelector('.hl-zone-interact');
    if (!zone) {
        shadow.innerHTML =
            '<div class="hl-zone-zoom"></div>'
            + '<div class="hl-zone-interact"></div>';
        zone = shadow.querySelector('.hl-zone-interact');
    }
    zone.innerHTML = '';

    /* ── Cancel any existing rAF loop ────────────────────── */
    const prev = host.__scoutState;
    if (prev && prev.rafId) {
        cancelAnimationFrame(prev.rafId);
        prev.running = false;
    }

    /* ── Initialize unified state ────────────────────────── */
    const state = {
        rafId: null,
        running: false,
        interactTracked: [],
        zoomTracked: prev ? prev.zoomTracked : [],
    };
    host.__scoutState = state;

    /* ── Inject styles ────────────────────────────────────── */
    const style = document.createElement('style');
    style.textContent = `
        /* Shared base — marching-ant pattern (same as zoom) */
        .hl-nav, .hl-mut, .hl-pass {
            position: fixed;
            pointer-events: none;
            border-radius: 6px;
            background-size: 2px 12px, 12px 2px, 2px 12px, 12px 2px;
            background-repeat: repeat-y, repeat-x, repeat-y, repeat-x;
            opacity: 0;
        }

        /* Navigating — orange/amber marching ants, self-removing */
        .hl-nav {
            background-color: rgba(245,158,11,0.10);
            background-image:
                linear-gradient(0deg,   rgba(245,158,11,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(245,158,11,0.50) 50%, transparent 50%),
                linear-gradient(0deg,   rgba(245,158,11,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(245,158,11,0.50) 50%, transparent 50%);
            box-shadow: inset 0 0 30px rgba(245,158,11,0.06),
                        0 0 12px rgba(245,158,11,0.12);
        }
        /* Mutating — purple marching ants, persistent */
        .hl-mut {
            background-color: rgba(139,92,246,0.10);
            background-image:
                linear-gradient(0deg,   rgba(139,92,246,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(139,92,246,0.50) 50%, transparent 50%),
                linear-gradient(0deg,   rgba(139,92,246,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(139,92,246,0.50) 50%, transparent 50%);
            box-shadow: inset 0 0 30px rgba(139,92,246,0.06),
                        0 0 12px rgba(139,92,246,0.12);
        }
        /* Passive — teal marching ants, persistent */
        .hl-pass {
            background-color: rgba(20,184,166,0.10);
            background-image:
                linear-gradient(0deg,   rgba(20,184,166,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(20,184,166,0.50) 50%, transparent 50%),
                linear-gradient(0deg,   rgba(20,184,166,0.50) 50%, transparent 50%),
                linear-gradient(90deg,  rgba(20,184,166,0.50) 50%, transparent 50%);
            box-shadow: inset 0 0 30px rgba(20,184,166,0.06),
                        0 0 12px rgba(20,184,166,0.12);
        }

        /* Animations */
        @keyframes scout-march-interact {
            0%   { background-position: 0 0, 0 0, 100% 0, 0 100%; }
            100% { background-position: 0 -24px, 24px 0, 100% 24px, -24px 100%; }
        }
        @keyframes scout-nav-pulse {
            0%   { opacity: 1; }
            100% { opacity: 0; }
        }
        @keyframes scout-interact-in {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes scout-interact-cursor {
            from { border-right-color: rgba(255,255,255,0.8); }
            to   { border-right-color: transparent; }
        }

        /* Labels — pinned to top-left edge, same style as zoom labels */
        .hl-interact-label {
            position: absolute;
            top: -1px;
            left: 8px;
            transform: translateY(-100%);
            padding: 3px 10px;
            border-radius: 4px 4px 0 0;
            font: 500 11px/1.2 -apple-system, BlinkMacSystemFont,
                  "Segoe UI", Roboto, sans-serif;
            color: rgba(255,255,255,0.92);
            letter-spacing: 0.3px;
            white-space: nowrap;
            overflow: hidden;
            border-right: 2px solid rgba(255,255,255,0.8);
            animation: scout-interact-cursor 0.55s step-end infinite;
        }
        .hl-nav .hl-interact-label  { background: rgba(180,83,9,0.75); }
        .hl-mut .hl-interact-label  { background: rgba(109,40,217,0.75); }
        .hl-pass .hl-interact-label { background: rgba(13,148,136,0.75); }

        /* Count badge for query_selector_all */
        .hl-count-badge {
            position: absolute; top: -8px; right: -8px;
            padding: 2px 6px; border-radius: 8px;
            background: rgba(13,148,136,0.85);
            font: 600 10px/1.2 -apple-system, sans-serif;
            color: #fff; pointer-events: none;
        }
    `;
    zone.appendChild(style);

    /* ── Label map ────────────────────────────────────────── */
    const LABELS = {
        click: 'Clicking\u2026', dblclick: 'Clicking\u2026', tap: 'Tapping\u2026',
        fill: 'Typing\u2026', type: 'Typing\u2026', press: 'Pressing key\u2026',
        press_sequentially: 'Typing\u2026',
        check: 'Checking\u2026', uncheck: 'Unchecking\u2026',
        set_checked: 'Checking\u2026', select_option: 'Selecting\u2026',
        query_selector: 'Reading\u2026', query_selector_all: 'Reading\u2026',
        inner_text: 'Reading text\u2026', text_content: 'Reading text\u2026',
        inner_html: 'Reading\u2026', get_attribute: 'Reading\u2026',
        evaluate: 'Reading\u2026', wait_for_selector: 'Waiting\u2026',
        input_value: 'Reading\u2026',
    };

    /* ── Helper: find first in-viewport element ─────────── */
    const vpW = window.innerWidth, vpH = window.innerHeight;
    function findVisible(selector) {
        try {
            const all = document.querySelectorAll(selector);
            for (const el of all) {
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) continue;
                if (r.bottom > 0 && r.top < vpH && r.right > 0 && r.left < vpW) {
                    return { x: r.left, y: r.top, width: r.width, height: r.height };
                }
            }
            for (const el of all) {
                const r = el.getBoundingClientRect();
                if (r.width > 0 || r.height > 0) {
                    return { x: r.left, y: r.top, width: r.width, height: r.height };
                }
            }
        } catch(e) { /* invalid selector */ }
        return null;
    }

    /* ── Pass 1: resolve all bounding rects ──────────────── */
    const resolved = [];
    for (const item of items) {
        try {
            let rect = item.rect || null;
            if (!rect && item.selector) {
                rect = findVisible(item.selector);
            }
            if (!rect || (rect.width === 0 && rect.height === 0)) continue;
            resolved.push({ item, rect });
        } catch(e) { /* skip */ }
    }

    /* ── Pass 2: overlap dedup ────────────────────────────────
     * Strategy: keep smaller, more specific overlays.
     *  - Sort smallest first, add to keep.
     *  - When considering a larger rect, count how many already-
     *    kept smaller rects it contains (≥90% of small area).
     *  - If it dominates >2 kept rects → drop it (the small ones
     *    are more informative than one giant overlay).
     *  - If it dominates ≤2 → drop the dominated small ones and
     *    keep the large one (avoids near-duplicate stacking).
     */
    function overlapArea(a, b) {
        const ox = Math.max(0, Math.min(a.x+a.width, b.x+b.width) - Math.max(a.x, b.x));
        const oy = Math.max(0, Math.min(a.y+a.height, b.y+b.height) - Math.max(a.y, b.y));
        return ox * oy;
    }
    resolved.sort((a, b) => (a.rect.width * a.rect.height) - (b.rect.width * b.rect.height));
    const keep = [];
    for (let i = 0; i < resolved.length; i++) {
        const big = resolved[i].rect;
        const bigArea = big.width * big.height;
        /* Find which kept rects this one dominates */
        const dominated = [];
        for (let j = 0; j < keep.length; j++) {
            const sm = keep[j].rect;
            const smArea = sm.width * sm.height;
            if (smArea < bigArea && overlapArea(sm, big) >= smArea * 0.9) {
                dominated.push(j);
            }
        }
        if (dominated.length > 2) {
            /* Large rect covers many small ones — drop the large */
            continue;
        }
        /* Remove the few dominated small rects, keep the large */
        for (let d = dominated.length - 1; d >= 0; d--) {
            keep.splice(dominated[d], 1);
        }
        keep.push(resolved[i]);
    }

    /* ── Pass 3: draw surviving overlays + build tracking ── */
    const tracked = [];

    for (let ki = 0; ki < keep.length; ki++) {
        const { item, rect } = keep[ki];
        try {
            const cat = item.category;
            const cls = cat === 'navigating' ? 'hl-nav'
                      : cat === 'mutating'   ? 'hl-mut'
                      : 'hl-pass';
            const d = document.createElement('div');
            d.className = cls;
            d.style.top    = rect.y + 'px';
            d.style.left   = rect.x + 'px';
            d.style.width  = rect.width  + 'px';
            d.style.height = rect.height + 'px';

            const delay = item.index * 100;

            /* Apply animation based on category */
            if (cat === 'navigating') {
                d.style.animation =
                    'scout-march-interact 0.4s linear ' + delay + 'ms infinite, '
                    + 'scout-nav-pulse 600ms ease-out ' + delay + 'ms forwards';
                /* Self-remove navigating overlays and clean up tracking */
                setTimeout(() => {
                    try { d.remove(); } catch(e) {}
                    const st = host.__scoutState;
                    if (st) st.interactTracked = st.interactTracked.filter(t => t.div !== d);
                }, delay + 650);
            } else {
                d.style.animation =
                    'scout-march-interact 0.4s linear ' + delay + 'ms infinite, '
                    + 'scout-interact-in 0.35s ease ' + delay + 'ms forwards';
            }

            /* Label */
            {
                const labelText = LABELS[item.action] || 'Interacting\u2026';
                const lbl = document.createElement('span');
                lbl.className = 'hl-interact-label';
                d.appendChild(lbl);
                let ci = 0;
                const startType = () => {
                    const typeChar = () => {
                        if (ci < labelText.length) {
                            lbl.textContent += labelText[ci++];
                            setTimeout(typeChar, 30 + Math.random() * 20);
                        }
                    };
                    typeChar();
                };
                setTimeout(startType, delay);
            }

            /* Count badge for query_selector_all */
            if (item.totalCount > 0 && item.index === 0) {
                const badge = document.createElement('span');
                badge.className = 'hl-count-badge';
                badge.textContent = item.totalCount + ' elements';
                d.appendChild(badge);
            }

            zone.appendChild(d);

            /* Track for rAF position updates */
            const sel = item.trackSelector || item.selector;
            if (sel) {
                tracked.push({ selector: sel, div: d });
            }
        } catch(e) { /* skip silently */ }
    }

    /* ── Store tracking state + start rAF loop ───────────── */
    state.interactTracked = tracked;

    function updatePositions() {
        const h = document.getElementById('scout-hl-host');
        if (!h || !h.__scoutState || !h.__scoutState.running) return;
        const st = h.__scoutState;

        for (const arr of [st.interactTracked, st.zoomTracked]) {
            for (const entry of arr) {
                try {
                    const el = document.querySelector(entry.selector);
                    if (!el) { entry.div.style.display = 'none'; continue; }
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) {
                        entry.div.style.display = 'none';
                        continue;
                    }
                    entry.div.style.display = '';
                    entry.div.style.top    = r.top + 'px';
                    entry.div.style.left   = r.left + 'px';
                    entry.div.style.width  = r.width + 'px';
                    entry.div.style.height = r.height + 'px';
                } catch(e) { entry.div.style.display = 'none'; }
            }
        }

        st.rafId = requestAnimationFrame(updatePositions);
    }

    state.running = true;
    state.rafId = requestAnimationFrame(updatePositions);

    /* ── Build per-item status array ─────────────────────── */
    const keptSet = new Set(keep.map(k => k.item.index));
    const statuses = items.map(item => {
        const wasResolved = resolved.some(r => r.item.index === item.index);
        if (!wasResolved) return 'not_found';
        if (keptSet.has(item.index)) return 'drawn';
        return 'overlap';
    });

    return { resolved: resolved.length, kept: keep.length, dropped: resolved.length - keep.length, statuses };
}"""
