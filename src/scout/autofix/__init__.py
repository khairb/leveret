"""Auto-fix system for detecting and regenerating stale scraping scripts.

Public API
----------
- ``diagnose()`` — orchestrates 3-attempt diagnosis loop
- ``AutoFixMode`` — user-selected risk tolerance enum
- ``DiagnosisResult`` — diagnosis output with action and evidence
- ``AttemptResult`` — result of a single script execution attempt

All other modules (classifier, fingerprint, antibot, page_verifier,
stability, decision) are implementation details.

Spec reference: docs/specific/AUTO_FIX_ALGORITHM.md
"""

from scout.autofix.diagnosis import diagnose
from scout.autofix.types import AttemptResult, AutoFixMode, DiagnosisResult

__all__ = [
    "AttemptResult",
    "AutoFixMode",
    "DiagnosisResult",
    "diagnose",
]
