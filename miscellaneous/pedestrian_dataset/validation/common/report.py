"""
Report writer shared by all validators.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, List, Optional

import yaml

from .results import SectionResult, rollup

logger = logging.getLogger(__name__)

_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"
_COLOR = {"pass": "\033[32m", "warn": "\033[33m", "fail": "\033[31m", "skip": "\033[90m"}
_ICON = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "•"}


def overall_status(sections: List[SectionResult]) -> str:
    return rollup([s.status for s in sections])


def _c(text: str, status: str) -> str:
    return f"{_COLOR.get(status, '')}{text}{_RESET}"


def _sep(char: str = "─", width: int = 74) -> None:
    print(_DIM + char * width + _RESET)


def print_report(title: str, sections: List[SectionResult],
                 context: Optional[Dict[str, str]] = None,
                 show_passes: bool = False) -> None:
    """
    Print a colored summary. By default only warn/fail/skip checks are listed in
    the per-section detail (pass-heavy output is noise); set show_passes=True to
    list everything.
    """
    print()
    _sep("═")
    print(f"{_BOLD}  {title}{_RESET}")
    for k, v in (context or {}).items():
        print(f"  {k:<26}: {v}")
    _sep("═")

    print(f"\n  {'Section':<34}{'Status':<8}{'checks (✓/⚠/✗/•)':>22}")
    _sep()
    for s in sections:
        c = s.counts()
        summary = f"{c['pass']}/{c['warn']}/{c['fail']}/{c['skip']}"
        print(f"  {s.name:<34}{_c(_ICON[s.status] + ' ' + s.status.upper(), s.status):<8}"
              f"{summary:>22}")
    _sep()
    ov = overall_status(sections)
    print(f"\n  Overall: {_c(_BOLD + _ICON[ov] + ' ' + ov.upper() + _RESET, ov)}\n")

    for s in sections:
        shown = [c for c in s.checks if show_passes or c.status != "pass"]
        if not shown and not s.figures:
            continue
        _sep("─")
        print(f"{_BOLD}  {s.name}{_RESET}  ({_c(s.status.upper(), s.status)})")
        _sep("─")
        for c in shown:
            print(f"  {_c(_ICON[c.status], c.status)}  {c.name}")
            print(f"     {_DIM}{c.message}{_RESET}")
        if not shown:
            print(f"  {_DIM}(all {len(s.checks)} checks passed){_RESET}")
        if s.figures:
            print(f"\n  {_DIM}figures:{_RESET}")
            for f in s.figures:
                print(f"    {_DIM}{f}{_RESET}")
    print()
    _sep("═")
    print()


def write_yaml_report(title: str, sections: List[SectionResult], out_dir: str,
                      filename: str, context: Optional[Dict[str, str]] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "title": title,
        "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_status": overall_status(sections),
        "context": context or {},
        "sections": [
            {
                "name": s.name,
                "status": s.status,
                "counts": s.counts(),
                "figures": s.figures,
                "checks": [
                    {"name": c.name, "status": c.status, "message": c.message,
                     **({"value": c.value} if c.value is not None else {})}
                    for c in s.checks
                ],
            }
            for s in sections
        ],
    }
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as fh:
        yaml.dump(report, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Report written: %s", out_path)
    return out_path