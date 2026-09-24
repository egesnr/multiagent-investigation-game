"""
Does the investigator actually use the case it was given?

Two measures, both read off a finished playtest log, both objective:

  fact coverage   — of the facts the investigator can see, how many did it
                    ever put to the suspect? A run that ends with half the
                    file unused was not an interview. The ege5 baseline spent
                    three of eight turns on the restaurant's name and never
                    raised the spending cap, the colleague's statement or the
                    missing receipt.

  inferences      — the conclusions that need two pieces of the case at once
                    (an amount against a cap, a date against an itinerary).
                    These are the ones a split pipeline structurally cannot
                    reach, because no single agent held both pieces.

Matching is keyword-based and deliberately generous: it answers "was this ever
raised at all", not "was it raised well". Overcounting is the safe direction —
if coverage still looks low under a generous matcher, it really is low.

Usage:
    python eval_coverage.py playtest_out/ege5.log playtest_out/ege_merged.log
"""

import argparse
import json
import re
from pathlib import Path

from main import DEFAULT_CASE_PATH, load_case

# What counts as having raised each fact. Keyed by fact id so a new case file
# only needs its own entry here, and an unlisted fact falls back to keywords
# pulled from its own description.
FACT_CUES: dict[str, list[str]] = {
    "solo_dinner": ["alone", "by yourself", "one person", "single diner", "table for one"],
    "no_calendar_entry": ["calendar", "diary", "schedule"],
    "actual_amount": ["3,200", "3200", "amount", "charge"],
    "no_supporting_documentation": ["receipt", "itemized", "documentation", "exception approval"],
    "no_prior_issue_report": ["report", "finance", "flag", "dispute", "notify"],
    "flagged_by_routine_audit": ["audit", "routine", "quarterly", "three weeks", "surfaced"],
    "trip_approved": ["trip", "stockholm", "summit", "approved"],
    "dinner_within_trip_window": ["september 4", "sept 4", "the 4th", "itinerary", "that evening"],
    "colleague_on_trip": ["colleague", "ravi", "chen", "anyone else", "who else"],
}

# The conclusions that need two parts of the case held at once. Each is a list
# of alternative cue sets; any one set matching counts as raised.
INFERENCE_CUES: dict[str, list[list[str]]] = {
    "amount vs the $75 cap": [["cap"], ["75"], ["limit", "meal"]],
    "no colleague corroboration (Ravi)": [["ravi"], ["colleague"], ["anyone else", "trip"]],
    "receipt mandatory over $100": [["itemized receipt"], ["receipt", "required"], ["receipt", "policy"]],
    "no business activity on Sept 4": [["itinerary"], ["no business", "that evening"], ["nothing scheduled"]],
    "24-hour reporting vs three weeks": [["24 hour"], ["24-hour"], ["within a day"], ["three weeks", "report"]],
    "found by audit, not self-reported": [["not self-reported"], ["you didn't report"], ["routine review"], ["audit found"]],
    "concealment once aware": [["concealment"], ["once you knew"], ["knew and said nothing"], ["three weeks"]],
}


def investigator_lines(log_text: str) -> list[str]:
    """The lines actually spoken to the suspect."""
    return [m.group(1).strip() for m in re.finditer(r"^<<< (.+)$", log_text, re.M)]


def _fallback_cues(description: str) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", description)]
    return words[:4]


def score_log(path: Path, case) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    spoken = " ".join(investigator_lines(text)).lower()

    visible = case.visible_facts("investigator_start")
    raised, missed = [], []
    for fact in visible:
        cues = FACT_CUES.get(fact.id) or _fallback_cues(fact.description)
        (raised if any(c in spoken for c in cues) else missed).append(fact.id)

    hit, absent = [], []
    for name, cue_sets in INFERENCE_CUES.items():
        ok = any(all(c in spoken for c in cue_set) for cue_set in cue_sets)
        (hit if ok else absent).append(name)

    turns = re.findall(r"\[turn\] ([+-]?\d+) \| score=(\d+)", text)
    return {
        "log": path.name,
        "questions_asked": len(spoken and investigator_lines(text) or []),
        "fact_coverage": f"{len(raised)}/{len(visible)}",
        "facts_raised": raised,
        "facts_never_raised": missed,
        "inferences_made": hit,
        "inferences_missed": absent,
        "final_score": int(turns[-1][1]) if turns else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE_PATH)
    args = parser.parse_args()

    case = load_case(args.case)
    for log in args.logs:
        print(json.dumps(score_log(log, case), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
