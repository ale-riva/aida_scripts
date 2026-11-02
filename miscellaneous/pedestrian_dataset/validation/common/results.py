"""
Shared result types for every validator (intermediate, KITTI, future nuScenes).

Status values
    pass: check satisfied
    warn: non-blocking issue (degraded but usable)
    fail: blocking issue (the dataset is wrong/unusable for this aspect)
    skip: check not applicable (e.g. no labels present, no IR to compare to)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# pass/skip are neutral; warn/fail escalate. Used to roll a section/overall up
# to its worst contained status.
_RANK: Dict[str, int] = {"pass": 0, "skip": 0, "warn": 1, "fail": 2}


def worst(a: str, b: str) -> str:
    """Return the more severe of two statuses."""
    return a if _RANK.get(a, 0) >= _RANK.get(b, 0) else b


def rollup(statuses: List[str]) -> str:
    """Return the worst status in a list (pass if empty)."""
    out = "pass"
    for s in statuses:
        out = worst(out, s)
    return out


@dataclass
class CheckResult:
    """One atomic validation check."""
    name: str
    status: str                       # pass | warn | fail | skip
    message: str
    value: Optional[float] = None      # numeric value when meaningful (for the YAML report)


@dataclass
class SectionResult:
    """Aggregated result for one check module (e.g. 'timestamps', 'calibration')."""
    name: str
    status: str = "pass"
    checks: List[CheckResult] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)   # absolute paths to saved PNGs

    # -- check recording -------------------------------------------------
    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        self.status = worst(self.status, check.status)

    def ok(self, name: str, message: str, value: Optional[float] = None) -> None:
        self.add(CheckResult(name, "pass", message, value))

    def warn(self, name: str, message: str, value: Optional[float] = None) -> None:
        self.add(CheckResult(name, "warn", message, value))

    def fail(self, name: str, message: str, value: Optional[float] = None) -> None:
        self.add(CheckResult(name, "fail", message, value))

    def skip(self, name: str, message: str) -> None:
        self.add(CheckResult(name, "skip", message))

    # -- figures ---------------------------------------------------------
    def add_figure(self, path: Optional[str]) -> None:
        if path:
            self.figures.append(path)

    # -- summary ---------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        out = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out