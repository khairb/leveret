"""Decision engine — routes all signals to a REGENERATE or RAISE action.

Takes the error category, stability level, page verification results,
and auto-fix mode, and produces a decision with a human-readable reason.

Decision flow (spec §9):
  1. Immediate decisions: A→REGENERATE, F2/F3/C/F1→RAISE, E-ineligible→RAISE
  2. Stability gate: MIXED/CHAOTIC → always RAISE
  3. Universal page checks: ANTI_BOT blocks all, <=1/3 REAL_PAGE blocks all
  4. Tier-specific decision tables (§9 matrices are authoritative)

Note on page verification gates:
  The §6 "universal gate" (cautious/balanced = 3/3 REAL_PAGE) is a
  simplified description. The §9 decision matrices are the authoritative
  source — they show that balanced mode can regenerate at 2/3 REAL_PAGE
  for Tier 1 (B, G) when stability is STABLE. The decision engine
  implements the §9 tables directly.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md §9
"""

from __future__ import annotations

from scout.autofix.types import (
    TIER_3_CATEGORIES,
    AutoFixAction,
    ErrorCategory,
    PageVerificationResult,
    RegenerateMode,
    StabilityLevel,
)

# ── Public API ───────────────────────────────────────────────


def decide(
    category: ErrorCategory,
    stability: StabilityLevel | None,
    page_results: list[PageVerificationResult],
    mode: RegenerateMode,
    e_eligible: bool = True,
) -> tuple[AutoFixAction, str]:
    """Decide whether to regenerate or raise based on all diagnostic signals.

    Implements the complete decision matrix from spec §9.

    Args:
        category: Error category from the classifier.
        stability: Stability assessment across failed attempts.
            None for categories that skip retries (A, F2, F3).
        page_results: Page verification results from each attempt.
            Empty for categories where page verification is N/A.
        mode: User's auto-fix risk tolerance.
        e_eligible: Whether Category E errors are eligible for regeneration.
            Only meaningful when category is E. True by default.

    Returns:
        Tuple of ``(action, reason_string)``.
        ``action`` is REGENERATE or RAISE.
        ``reason_string`` explains the decision for logging/error messages.
    """
    # ── 1. Immediate decisions (no stability or page needed) ──

    # §9: Category A → REGENERATE immediately (100% script fault)
    if category == ErrorCategory.A:
        return (
            AutoFixAction.REGENERATE,
            "Parse error (Category A) — code is structurally broken, regenerating immediately",
        )

    # §9: Tier 3 — never regenerate (C, F1, F2, F3)
    if category in TIER_3_CATEGORIES:
        return _raise_tier3(category)

    # §9: Category E context/frame destruction — never eligible
    if category == ErrorCategory.E and not e_eligible:
        return (
            AutoFixAction.RAISE,
            "Page state error (Category E) — context/frame destruction "
            "is not eligible for regeneration in any mode",
        )

    # ── 2. Stability gate ──
    # MIXED or CHAOTIC → always RAISE, regardless of other signals

    if stability in (StabilityLevel.MIXED, StabilityLevel.CHAOTIC):
        return (
            AutoFixAction.RAISE,
            f"Failure pattern is {stability.value} — evidence too noisy to justify regeneration",
        )

    # ── 3. Universal page checks ──
    # §9: ANTI_BOT blocks all modes, all tiers.
    # §9: <=1/3 REAL_PAGE blocks all modes, all tiers.
    # These are checked before tier-specific logic.

    page_block = _check_universal_page_rules(page_results)
    if page_block is not None:
        return page_block

    # ── 4. Tier-specific decisions ──
    # At this point: stability is STABLE or CONSISTENT,
    # and universal page rules passed (≥2/3 REAL_PAGE, no ANTI_BOT).

    # Defensive: should never reach here with None/MIXED/CHAOTIC — the
    # stability gate above blocks those. But if it does (e.g., python -O
    # or future code changes), fail safe rather than silently producing
    # a wrong decision that spends the user's money.
    if stability not in (StabilityLevel.STABLE, StabilityLevel.CONSISTENT):
        return (
            AutoFixAction.RAISE,
            f"Unexpected stability {stability!r} after gate — defaulting to raise",
        )

    if category in (ErrorCategory.B, ErrorCategory.G):
        return _decide_tier1(stability, page_results, mode)

    if category in (ErrorCategory.D, ErrorCategory.E):
        return _decide_tier2(stability, page_results, mode)

    # Shouldn't reach here — all categories are handled above.
    # But defensive: treat unknown as RAISE.
    return (
        AutoFixAction.RAISE,
        f"Unknown category {category.value} — defaulting to raise",
    )


# ── Tier 1: High-confidence (B, G) ──────────────────────────


def _decide_tier1(
    stability: StabilityLevel,
    page_results: list[PageVerificationResult],
    mode: RegenerateMode,
) -> tuple[AutoFixAction, str]:
    """Decision table for Tier 1 categories (B, G).

    Spec §9 matrix:
      STABLE + 3/3 real → Regen (all modes)
      STABLE + 2/3 real → Raise (cautious), Regen (balanced/eager)
      CONSISTENT + 3/3 real → Regen (all modes)
      CONSISTENT + 2/3 real → Raise (cautious/balanced), Regen (eager)
    """
    real_count = _count_real(page_results)
    total = len(page_results)

    if stability == StabilityLevel.STABLE:
        if real_count == total:
            # STABLE + all real → Regen in all modes
            return (
                AutoFixAction.REGENERATE,
                f"Script fault confirmed — stable failure pattern, "
                f"page verified real ({real_count}/{total})",
            )
        # STABLE + partial real (gate already passed, so ≥2/3)
        if mode in (RegenerateMode.BALANCED, RegenerateMode.EAGER):
            return (
                AutoFixAction.REGENERATE,
                f"Script fault likely — stable failure pattern, "
                f"page mostly verified ({real_count}/{total}, {mode.value} mode)",
            )
        # Conservative: partial real is not enough
        return (
            AutoFixAction.RAISE,
            f"Page not fully verified ({real_count}/{total}) — "
            f"cautious mode requires all attempts verified real",
        )

    # CONSISTENT
    if real_count == total:
        # CONSISTENT + all real → Regen in all modes
        return (
            AutoFixAction.REGENERATE,
            f"Script fault likely — consistent failure pattern, "
            f"page verified real ({real_count}/{total})",
        )
    # CONSISTENT + partial real (gate already passed, so ≥2/3)
    if mode == RegenerateMode.EAGER:
        return (
            AutoFixAction.REGENERATE,
            f"Script fault possible — consistent failure pattern, "
            f"page mostly verified ({real_count}/{total}, eager mode)",
        )
    # Conservative/balanced: CONSISTENT + partial real → not enough
    return (
        AutoFixAction.RAISE,
        f"Consistent failure pattern but page not fully verified "
        f"({real_count}/{total}) — {mode.value} mode requires stronger evidence",
    )


# ── Tier 2: Ambiguous (D, E-eligible) ───────────────────────


def _decide_tier2(
    stability: StabilityLevel,
    page_results: list[PageVerificationResult],
    mode: RegenerateMode,
) -> tuple[AutoFixAction, str]:
    """Decision table for Tier 2 categories (D, E eligible).

    Spec §9 matrix:
      STABLE + 3/3 real → Raise (cautious), Regen (balanced/eager)
      STABLE + 2/3 real → Raise (cautious/balanced), Regen (eager)
      CONSISTENT + 3/3 real → Raise (cautious/balanced), Regen (eager)
      CONSISTENT + 2/3 real → Raise (all modes)
    """
    real_count = _count_real(page_results)
    total = len(page_results)

    if stability == StabilityLevel.STABLE:
        if real_count == total:
            # STABLE + all real
            if mode == RegenerateMode.CAUTIOUS:
                return (
                    AutoFixAction.RAISE,
                    f"Timeout/state error with stable pattern and verified real page "
                    f"({real_count}/{total}) — cautious mode does not regenerate "
                    f"ambiguous categories",
                )
            # Balanced or eager
            return (
                AutoFixAction.REGENERATE,
                f"Timeout/state error — stable failure on verified real page "
                f"({real_count}/{total}, {mode.value} mode)",
            )
        # STABLE + partial real (gate already passed, so ≥2/3)
        if mode == RegenerateMode.EAGER:
            return (
                AutoFixAction.REGENERATE,
                f"Timeout/state error — stable failure, page mostly verified "
                f"({real_count}/{total}, eager mode)",
            )
        return (
            AutoFixAction.RAISE,
            f"Timeout/state error with partial page verification "
            f"({real_count}/{total}) — {mode.value} mode requires stronger evidence",
        )

    # CONSISTENT
    if real_count == total:
        if mode == RegenerateMode.EAGER:
            return (
                AutoFixAction.REGENERATE,
                f"Timeout/state error — consistent failure on verified real page "
                f"({real_count}/{total}, eager mode)",
            )
        return (
            AutoFixAction.RAISE,
            f"Timeout/state error with consistent pattern and verified real page "
            f"({real_count}/{total}) — {mode.value} mode requires stable pattern "
            f"for ambiguous categories",
        )
    # CONSISTENT + partial real → Raise in all modes
    return (
        AutoFixAction.RAISE,
        f"Timeout/state error with consistent pattern but partial page "
        f"verification ({real_count}/{total}) — insufficient evidence",
    )


# ── Tier 3: Never regenerate ────────────────────────────────


def _raise_tier3(
    category: ErrorCategory,
) -> tuple[AutoFixAction, str]:
    """Produce RAISE decision for Tier 3 categories (C, F1, F2, F3)."""
    reasons = {
        ErrorCategory.C: (
            "Network/server failure (Category C) — page never loaded, "
            "a new script needs the same server to be up"
        ),
        ErrorCategory.F1: (
            "Browser/page crash (Category F1) — a new script runs in the "
            "same browser environment and will likely crash too"
        ),
        ErrorCategory.F2: (
            "Subprocess timeout (Category F2) — diagnostic cost prohibitive, not retrying"
        ),
        ErrorCategory.F3: (
            "Infrastructure failure (Category F3) — no script can execute "
            "until the infrastructure is repaired"
        ),
    }
    return AutoFixAction.RAISE, reasons[category]


# ── Helpers ──────────────────────────────────────────────────


def _check_universal_page_rules(
    page_results: list[PageVerificationResult],
) -> tuple[AutoFixAction, str] | None:
    """Check page rules that apply across all tiers and modes.

    §9: ANTI_BOT in any attempt → RAISE (all modes, all tiers).
    §9: <=1/3 REAL_PAGE → RAISE (all modes, all tiers).
    §9: No page results → RAISE.

    Returns (RAISE, reason) if blocked, None if the tier-specific
    logic should proceed.
    """
    if not page_results:
        return (
            AutoFixAction.RAISE,
            "No page verification results available",
        )

    total = len(page_results)
    real_count = _count_real(page_results)
    has_antibot = any(r == PageVerificationResult.ANTI_BOT for r in page_results)

    # §9: ANTI_BOT blocks all modes, even eager
    if has_antibot:
        antibot_count = sum(1 for r in page_results if r == PageVerificationResult.ANTI_BOT)
        return (
            AutoFixAction.RAISE,
            f"Anti-bot detected in {antibot_count}/{total} attempts — regeneration blocked",
        )

    # §9: <=1/3 REAL_PAGE blocks all modes
    # With 3 attempts: need at least 2 real. With 2: need at least 2.
    required_min = max(2, (total * 2 + 2) // 3)  # ceil(2/3 * total)
    if real_count < required_min:
        taint = _describe_taint(page_results)
        return (
            AutoFixAction.RAISE,
            f"Insufficient page verification ({real_count}/{total} REAL_PAGE"
            f"{taint}) — regeneration blocked",
        )

    return None


def _describe_taint(
    page_results: list[PageVerificationResult],
) -> str:
    """Describe non-REAL_PAGE results for error messages."""
    parts: list[str] = []
    for result_type in (
        PageVerificationResult.SERVER_ERROR,
        PageVerificationResult.SOFT_BLOCK,
        PageVerificationResult.REDIRECTED,
        PageVerificationResult.NO_RESPONSE,
    ):
        count = sum(1 for r in page_results if r == result_type)
        if count > 0:
            parts.append(f"{count}x {result_type.value}")
    if parts:
        return ", " + ", ".join(parts)
    return ""


def _count_real(page_results: list[PageVerificationResult]) -> int:
    """Count REAL_PAGE results."""
    return sum(1 for r in page_results if r == PageVerificationResult.REAL_PAGE)
