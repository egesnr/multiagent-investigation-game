"""
CLI orchestration for the multi-agent investigation game.
"""

import json

from models import CaseFile, GameState
from game_logic import (
    apply_tier,
    case_decisively_resolved,
    dedupe_results,
    process_turn_scoring,
)
from agents import run_checker, run_extractor, run_speaker, run_strategist
from resolution import format_resolution, run_resolution


DEFAULT_CASE_PATH = "case_files/restaurant_case.json"


def load_case(path: str = DEFAULT_CASE_PATH) -> CaseFile:
    with open(path, encoding="utf-8") as f:
        return CaseFile(**json.load(f))


def opening_question(case: CaseFile) -> str:
    visible = case.visible_facts("investigator_start")
    amount_fact = next(
        (
            f for f in visible
            if "amount" in f.id.lower()
            or "transaction" in f.description.lower()
            or "charge" in f.description.lower()
        ),
        None,
    )

    if amount_fact:
        return (
            "Walk me through this expense from the beginning. "
            f"I want you to explain the {amount_fact.true_value} transaction."
        )
    return f"Walk me through what happened in this {case.scenario_type} review."


def format_player_briefing(case: CaseFile) -> str:
    """
    Suspect-facing briefing for CLI play.

    The player knows their character's authored reality, but is not told the
    investigator's exact evidence state.
    """
    suspect_facts = case.visible_facts("suspect")
    fact_lines = "\n".join(
        f"- {f.description}: {f.true_value}"
        for f in suspect_facts
    ) or "- No additional suspect-only facts are defined."

    return f"""=== CASE BRIEFING ===

You are the employee being questioned in a {case.scenario_type} investigation.

What you know happened:
{fact_lines}

Your objective:
Avoid a final finding of intentional fraud.

You may tell the truth, lie, mix truth with lies, evade, refuse to answer, or invent explanations. New claims do not automatically count as false, but they may create contradictions, credibility problems, or things investigators can try to verify later.

You do not know exactly what the investigator has already confirmed.

The investigator has a limited number of questions and will try to determine what most likely happened.
"""


def run_turn(
    state: GameState,
    question: str,
    player_answer: str,
) -> tuple[GameState, str]:
    state.transcript.append({"role": "investigator", "text": question})
    state.transcript.append({"role": "suspect", "text": player_answer})
    state.question_count += 1

    extracted = run_extractor(question, player_answer)

    for defense in extracted.new_defenses:
        defense = defense.strip()
        if defense and defense not in state.case_log.active_defenses:
            state.case_log.active_defenses.append(defense)

    results = run_checker(
        state.case,
        extracted.claims,
        transcript=state.transcript,
        case_log=state.case_log,
    )
    results = dedupe_results(results)

    turn_delta, state = process_turn_scoring(results, state, player_answer)

    for result in results:
        rp = result.risk_profile
        print(
            "  [checker] "
            f"basis={result.basis.value} | visible={result.investigator_visible} | "
            f"fact_contradiction={rp.fact_contradiction} | "
            f"story_contradiction={rp.story_contradiction} | "
            f"policy_breach={rp.policy_breach} | "
            f"proof_deficit={rp.proof_deficit} | "
            f"credibility_issue={rp.credibility_issue} — "
            f"\"{result.quoted_evidence}\""
        )

    print(
        f"  [turn] +{turn_delta} | score={state.score} | "
        f"questions={state.question_count}/{state.max_questions}"
    )

    if state.question_count >= state.max_questions:
        closing = (
            "That concludes the interview. Your statements will now be reviewed "
            "against the available evidence."
        )
        state.transcript.append({"role": "investigator", "text": closing})
        return state, closing

    if case_decisively_resolved(state):
        closing = "I have enough for now. This interview is concluded."
        state.transcript.append({"role": "investigator", "text": closing})
        return state, closing

    _, allowed_moves = apply_tier(state.score)
    move = run_strategist(state, allowed_moves)

    state.last_move = f"{move.tactic}:{move.target}"
    state.move_history.append(state.last_move)

    print(
        f"  [strategy] tactic={move.tactic} | target={move.target} | "
        f"remaining={state.max_questions - state.question_count}"
    )

    line = run_speaker(
        state.case,
        move,
        state.transcript,
        score=state.score,
        case_log=state.case_log,
    )
    state.transcript.append({"role": "investigator", "text": line})
    return state, line


def get_final_resolution(state: GameState) -> str:
    return format_resolution(run_resolution(state))


if __name__ == "__main__":
    case = load_case()
    state = GameState(case=case)

    print(format_player_briefing(case))
    question = opening_question(case)
    print(f"REVIEWER: {question}\n")

    while state.question_count < state.max_questions:
        answer = input("YOU: ")
        state, next_line = run_turn(state, question, answer)

        print(
            f"\n[score: {state.score} | tier: {state.tier} | "
            f"questions: {state.question_count}/{state.max_questions}]"
        )
        print(f"REVIEWER: {next_line}\n")
        question = next_line

        if state.question_count >= state.max_questions or case_decisively_resolved(state):
            break

    print("\n--- POST-INTERVIEW RESOLUTION ---")
    print(get_final_resolution(state))
