"""
Dev harness for testing the interview pipeline against real LLM calls without
the interactive CLI. Persists GameState as JSON so a session can be advanced
one turn at a time, resumed, or replayed with a scripted answer list.

Usage:
  Start a fresh session and ask the opening question:
    python playtest.py --new --state out/session.json

  Advance one turn with a specific answer:
    python playtest.py --state out/session.json --answer "I was at dinner with a client."

  Run a whole scripted playthrough in one process (answers from a file, one per line):
    python playtest.py --new --state out/session.json --script answers.txt

  Inspect a saved session without advancing it:
    python playtest.py --state out/session.json --show
"""

import argparse
import json
from pathlib import Path

from models import GameState
from main import (
    DEFAULT_CASE_PATH,
    format_player_briefing,
    load_case,
    opening_question,
    run_turn,
)


def save_state(state: GameState, path: Path, question: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pending_question": question, "state": state.model_dump()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path) -> tuple[GameState, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GameState(**payload["state"]), payload["pending_question"]


def show(state: GameState, question: str) -> None:
    print(f"score={state.score} tier={state.tier} "
          f"questions={state.question_count}/{state.max_questions} "
          f"stonewall={state.consecutive_stonewall}")
    print(f"exhausted_targets={state.case_log.exhausted_targets}")
    print(f"parked_threads={state.case_log.parked_threads}")
    print(f"move_history={state.move_history}")
    if state.narrative:
        print(f"narrative_summary: {state.narrative.summary}")
    print(f"\nNEXT QUESTION: {question}")


def one_turn(state: GameState, question: str, answer: str) -> tuple[GameState, str]:
    print(f"\n>>> Q: {question}\n>>> A: {answer}")
    state, next_line = run_turn(state, question, answer)
    print(f"score={state.score} (+{state.last_turn_delta}) | "
          f"questions={state.question_count}/{state.max_questions} | "
          f"stonewall={state.consecutive_stonewall}")
    if state.move_history:
        print(f"move: {state.move_history[-1]}")
    print(f"<<< {next_line}")
    return state, next_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--case", default=DEFAULT_CASE_PATH)
    parser.add_argument("--new", action="store_true", help="Start a fresh session, overwriting --state.")
    parser.add_argument("--answer", help="Single scripted answer for one turn.")
    parser.add_argument("--script", type=Path, help="File of answers, one per line, run in sequence.")
    parser.add_argument("--show", action="store_true", help="Print current state without advancing.")
    args = parser.parse_args()

    if args.new:
        case = load_case(args.case)
        state = GameState(case=case)
        question = opening_question(case)
        print(format_player_briefing(case))
        print(f"REVIEWER: {question}\n")
        save_state(state, args.state, question)
        if not args.answer and not args.script:
            return
    else:
        state, question = load_state(args.state)

    if args.show:
        show(state, question)
        return

    if args.script:
        answers = [
            line.strip() for line in args.script.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for answer in answers:
            if state.question_count >= state.max_questions:
                break
            state, question = one_turn(state, question, answer)
            # Persist after every turn, not just at the end: a 503/429 mid-script
            # must not lose already-completed turns.
            save_state(state, args.state, question)
        return

    if args.answer:
        state, question = one_turn(state, question, args.answer)
        save_state(state, args.state, question)
        return

    show(state, question)


if __name__ == "__main__":
    main()
