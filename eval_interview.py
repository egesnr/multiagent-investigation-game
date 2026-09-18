"""
Offline quality report for a saved playtest session.

Runs against the JSON written by playtest.py — no API calls, no cost, so it
can be re-run freely and used to compare two runs against each other.

It measures the things that actually went wrong in testing, rather than
asserting pass/fail on LLM output (which is not deterministic):

  repeated openings  - the "That's not what I asked." degeneration, where the
                       investigator opens several lines the same way
  repeated phrases   - any longer phrase reused verbatim across lines
  question ratio     - every line ending in "?" reads like a questionnaire
  length spread      - uniform line length is the other half of sounding robotic
  target overlap     - the Strategist chasing the same thread in new wording

Usage:
    python eval_interview.py playtest_out/session.json [more_sessions.json ...]
"""

import json
import re
import statistics
import sys
from pathlib import Path


def investigator_lines(state: dict) -> list[str]:
    """Each question appears twice in the transcript — once when the Speaker
    generates it at the end of a turn, and again as the question being
    answered at the start of the next. Keep first occurrences only."""
    seen: set[str] = set()
    lines = []
    for turn in state.get("transcript", []):
        if turn.get("role") != "investigator":
            continue
        text = turn.get("text", "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return lines


def _words(text: str) -> list[str]:
    return [w.strip(".,!?;:\"'—-").lower() for w in text.split() if w.strip()]


def repeated_openings(lines: list[str], window: int = 4) -> list[str]:
    """Lines that start with the same first few words as an earlier line."""
    seen: dict[str, int] = {}
    hits = []
    for line in lines:
        key = " ".join(_words(line)[:window])
        if not key:
            continue
        if key in seen:
            hits.append(key)
        seen[key] = seen.get(key, 0) + 1
    return hits


def repeated_phrases(lines: list[str], n: int = 6) -> list[str]:
    """Any n-word phrase used in more than one line."""
    phrase_lines: dict[str, set[int]] = {}
    for i, line in enumerate(lines):
        words = _words(line)
        for start in range(max(len(words) - n + 1, 0)):
            phrase = " ".join(words[start:start + n])
            phrase_lines.setdefault(phrase, set()).add(i)
    return sorted(p for p, idx in phrase_lines.items() if len(idx) > 1)


def question_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return sum(1 for line in lines if line.rstrip().endswith("?")) / len(lines)


def length_spread(lines: list[str]) -> tuple[float, float]:
    counts = [len(_words(line)) for line in lines]
    if len(counts) < 2:
        return (float(counts[0]) if counts else 0.0, 0.0)
    return (statistics.mean(counts), statistics.pstdev(counts))


def target_overlap(move_history: list[str]) -> list[tuple[int, int, float]]:
    """Pairs of Strategist targets that share most of their significant words."""
    sets = [{w for w in _words(m) if len(w) > 3} for m in move_history]
    hits = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if not sets[i] or not sets[j]:
                continue
            overlap = len(sets[i] & sets[j]) / min(len(sets[i]), len(sets[j]))
            if overlap > 0.5:
                hits.append((i + 1, j + 1, round(overlap, 2)))
    return hits


def unsourced_claims(state: dict) -> list[str]:
    """Grounding entries that cite a known fact which does not exist.

    The Speaker names a source for each specific claim it makes, but nothing
    stops it inventing the source along with the claim — observed live, it
    asserted "no follow-up email or meeting notes in the three weeks after
    the dinner" and labelled that "known fact" when no such fact is authored.
    This compares each such citation against the real investigator-visible
    facts. Word overlap is deliberately coarse: this only flags entries for a
    human to read, it never changes what the game does.
    """
    facts = [
        f"{f.get('description','')} {f.get('true_value','')}"
        for f in state.get("case", {}).get("facts", [])
        if "investigator_start" in f.get("visible_to", [])
    ]
    fact_word_sets = [{w for w in _words(f) if len(w) > 3} for f in facts]

    suspicious = []
    for turn in state.get("grounding_history", []):
        for entry in turn:
            if "known fact" not in entry.lower():
                continue
            claim_words = {w for w in _words(entry) if len(w) > 3}
            claim_words -= {"known", "fact"}
            if not claim_words:
                continue
            best = max(
                (len(claim_words & fw) / len(claim_words) for fw in fact_word_sets),
                default=0.0,
            )
            if best < 0.5:
                suspicious.append(f"{entry}  [best match {best:.0%}]")
    return suspicious


_NUMERIC = re.compile(r"\b\d[\d,:.]*\b")


def uncited_specifics(state: dict) -> list[str]:
    """Numbers and times in a spoken line that appear nowhere in its sources.

    The Speaker names its sources before writing, but nothing forces the line
    to stay inside them. Observed live: it cited three genuine known facts and
    then told the suspect the charge landed "at 8:14 PM", a time that exists
    nowhere in the case. Citations were real; the sentence still invented a
    detail. Numbers are checked because they are unambiguous and are exactly
    where fabricated precision shows up.
    """
    facts_blob = " ".join(
        f"{f.get('description','')} {f.get('true_value','')}"
        for f in state.get("case", {}).get("facts", [])
    )
    lines = investigator_lines(state)
    grounding = state.get("grounding_history", [])

    flagged = []
    for i, line in enumerate(lines):
        sources = " ".join(grounding[i]) if i < len(grounding) else ""
        allowed = f"{sources} {facts_blob}"
        allowed_nums = set(_NUMERIC.findall(allowed.replace(",", "")))
        for token in set(_NUMERIC.findall(line.replace(",", ""))):
            if len(token) < 3:
                continue  # skip small counts like "1" or "20" that recur harmlessly
            if token not in allowed_nums:
                flagged.append(f'turn {i + 1}: "{token}" not in any cited source')
    return flagged


def report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["state"]
    lines = investigator_lines(state)
    mean_len, spread = length_spread(lines)

    print(f"\n=== {path.name} ===")
    print(f"investigator lines : {len(lines)}")
    print(f"final score        : {state.get('score')}")
    print(f"repeated openings  : {len(repeated_openings(lines))} {repeated_openings(lines) or ''}")
    print(f"repeated phrases   : {len(repeated_phrases(lines))}")
    for phrase in repeated_phrases(lines)[:5]:
        print(f"                     - \"{phrase}\"")
    print(f"question ratio     : {question_ratio(lines):.0%} of lines end in '?'")
    print(f"line length        : mean {mean_len:.1f} words, spread {spread:.1f}")
    overlaps = target_overlap(state.get("move_history", []))
    print(f"similar targets    : {len(overlaps)} pair(s) {overlaps or ''}")
    bad = unsourced_claims(state)
    print(f"invented citations : {len(bad)}")
    for entry in bad:
        print(f"                     ! {entry}")
    loose = uncited_specifics(state)
    print(f"uncited numbers    : {len(loose)}")
    for entry in loose:
        print(f"                     ! {entry}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for arg in sys.argv[1:]:
        report(Path(arg))
