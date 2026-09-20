"""
CLI orchestration for the multi-agent investigation game.
"""

import json

from models import (
    CaseFile,
    CheckResult,
    ClaimNoveltyStatus,
    ClaimStatus,
    ClaimType,
    FindingBasis,
    GameState,
    RiskProfile,
)
from game_logic import (
    case_decisively_resolved,
    dedupe_results,
    process_turn_scoring,
)
from agents import (
    run_checker,
    run_extractor,
    run_narrative_synthesis,
    run_reality_gate,
    run_speaker,
    run_strategist,
)
from resolution import format_resolution, run_resolution


DEFAULT_CASE_PATH = "case_files/restaurant_case.json"


LOG_PATH = "game_debug_log.txt"


def _append_debug_log(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def reset_debug_log() -> None:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("MULTI-AGENT INVESTIGATION DEBUG LOG\n")
        f.write("=" * 60 + "\n\n")


def _format_checker_results(results) -> str:
    if not results:
        return "- No checker findings."

    lines = []
    for r in results:
        rp = r.risk_profile
        lines.append(
            f"- claim={r.quoted_evidence!r}\n"
            f"  basis={r.basis.value}, visible={r.investigator_visible}, "
            f"type={r.claim_type.value}, status={r.verification_status.value}, "
            f"ambiguous={r.ambiguous}\n"
            f"  strategic_value={r.strategic_value}, "
            f"future_verification_value={r.future_verification_value.value}, "
            f"evidentiary_impact={r.evidentiary_impact.value}\n"
            f"  contradiction={rp.fact_contradiction or rp.story_contradiction}, "
            f"policy_breach={rp.policy_breach}, credibility_issue={rp.credibility_issue}, "
            f"evasion={rp.evasion}\n"
            f"  suggested_thread={r.suggested_thread}\n"
            f"  rationale={r.rationale}"
        )
    return "\n".join(lines)


def _format_extracted_claims(items: list) -> str:
    """Show each extracted claim with its Rule 2/3/4/5 classification."""
    if not items:
        return "- No claims extracted."

    lines = []
    for c in items:
        lines.append(
            f"- \"{c.text}\" "
            f"[status={c.status.value}, is_defense={c.is_defense}, "
            f"checkable={c.checkable}, ambiguous={c.ambiguous}]"
        )
    return "\n".join(lines)


def load_case(path: str = DEFAULT_CASE_PATH) -> CaseFile:
    with open(path, encoding="utf-8") as f:
        return CaseFile(**json.load(f))


def opening_question(case: CaseFile) -> str:
    # Keyed off the shape of the data (a dollar-amount true_value) rather than
    # fuzzy keyword matching on the description text, which previously broke
    # silently: "charge" as a substring of "charged" matched an unrelated
    # fact once the case file's wording changed.
    visible = case.visible_facts("investigator_start")
    amount_fact = next(
        (f for f in visible if f.true_value.strip().startswith("$")),
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
    state.last_world_notice = None
    gate = run_reality_gate(state, player_answer)

    if gate.has_blocked_action and gate.blocked_message:
        state.last_world_notice = gate.blocked_message
        print(f"  [world] {gate.blocked_message}")

    # A pure impossible world action does not consume the investigator's question.
    # The player gets to answer the same question again.
    if gate.has_blocked_action and not gate.analysis_text.strip():
        _append_debug_log(
            f"BLOCKED PLAYER ACTION\n{'-' * 60}\n"
            f"INVESTIGATOR QUESTION:\n{question}\n\n"
            f"RAW PLAYER INPUT:\n{player_answer}\n\n"
            f"REALITY GATE:\n{json.dumps(gate.model_dump(), indent=2, ensure_ascii=False)}\n"
            f"QUESTION NOT CONSUMED\n" + "=" * 60
        )
        return state, question

    analysis_answer = gate.analysis_text.strip() or player_answer

    state.transcript.append({"role": "investigator", "text": question})
    state.transcript.append({"role": "suspect", "text": analysis_answer})
    state.question_count += 1

    # Rule 1 (Context Synthesis) needs the previous question, passed below.
    # Rule 3 (State Diffing) needs prior claims on record so the extractor can tag
    # each new item NEW / REITERATED / UPDATED instead of re-logging restatements.
    # Rule 5 (Literal Attribution) needs enough conversational context to resolve
    # ambiguous referents (e.g. "the investigation") correctly, rather than
    # guessing from the single most recent question alone. This mirrors the
    # context window already given to run_reality_gate.
    known_claims = [c.text for c in state.case_log.claims]
    extracted = run_extractor(
        question,
        analysis_answer,
        known_claims=known_claims,
        transcript=state.transcript,
    )

    # Hold the suspect's account as one story, separate from the atomic
    # claims list below. This is what catches a suspect contradicting their
    # OWN earlier words (walked-back denials, a detail that quietly changed)
    # and what notices the investigator itself circling the same topic
    # without progress — neither of which the per-claim Checker pipeline can
    # see, since it only ever compares one claim at a time against facts.
    narrative = run_narrative_synthesis(state)
    state.narrative = narrative

    if narrative.stale_thread:
        stale_note = narrative.stale_thread.strip()
        if stale_note and stale_note not in state.case_log.exhausted_targets:
            state.case_log.exhausted_targets.append(stale_note)

    # Safety net: exact-text duplicates must never reach the Checker regardless
    # of what novelty status the model assigned. This does not replace Rule 3
    # (semantic restatement detection still relies on the model), it only
    # guarantees the trivial case - identical wording - can never slip through
    # as NEW/UPDATED due to a model miss.
    _known_normalized = {" ".join(t.lower().split()) for t in known_claims}
    for item in extracted.claims:
        normalized = " ".join(item.text.lower().split())
        if normalized in _known_normalized and item.status != ClaimNoveltyStatus.REITERATED:
            item.status = ClaimNoveltyStatus.REITERATED

    # Reiterated claims add no new evidentiary information. Skip re-checking and
    # re-logging them so the case log and Strategist context stay signal-only.
    novel_items = [
        item for item in extracted.claims
        if item.status != ClaimNoveltyStatus.REITERATED
    ]

    # Rule 4 (Admission vs. Defense Separation): a claim can be BOTH a checkable
    # fact and a defense (e.g. "the charge was a waitress error"). Track it as a
    # defense for narrative context...
    for item in novel_items:
        if item.is_defense and item.text not in state.case_log.active_defenses:
            state.case_log.active_defenses.append(item.text)

    # ...pure narrative/emotional content with no factual content still goes here.
    for defense in extracted.new_defenses:
        defense = defense.strip()
        if defense and defense not in state.case_log.active_defenses:
            state.case_log.active_defenses.append(defense)

    # Fix for the "defense verification gap": previously only extracted.claims
    # (plain admissions) reached the Checker, so checkable defenses like a
    # "waitress error" theory were never actually verified against evidence.
    # Now anything checkable (defense or not) is sent through.
    checkable_items = [item for item in novel_items if item.checkable]
    checkable_claims = [item.text for item in checkable_items]

    # Rule 5 enforcement: carry the ambiguity flag through to the Checker so an
    # ambiguous claim can never be treated as a firm story-history contradiction
    # against another of the suspect's statements (see game_logic._effective_status).
    ambiguous_map = {item.text: item.ambiguous for item in checkable_items}

    # Same fix pattern as investigator_visible/hidden_truth: is_defense is the
    # Extractor's judgment call on the suspect's rhetorical intent, not
    # something Stage 2 should independently re-guess from the claim text
    # alone. Thread it through so claim_type can be forced from it in code.
    defense_map = {item.text: item.is_defense for item in checkable_items}

    results = run_checker(
        state.case,
        checkable_claims,
        transcript=state.transcript,
        case_log=state.case_log,
        ambiguous_map=ambiguous_map,
        defense_map=defense_map,
    )

    # Self-contradictions are a different kind of finding from everything the
    # Checker produces: the narrative synthesis already did the verification
    # (it compared two of the suspect's own statements directly against the
    # transcript), so this is NOT sent through the Checker to be re-verified
    # against evidence — that would mean asking "is it true that the suspect
    # contradicted themselves" as if it were still an open question. Built
    # directly instead, with basis/status/claim_type forced by construction
    # rather than re-guessed.
    for contradiction in narrative.self_contradictions:
        results.append(CheckResult(
            quoted_evidence=contradiction.claim_text,
            rationale=(
                f"Earlier: \"{contradiction.earlier_statement}\" — "
                f"Later: \"{contradiction.later_statement}\""
            ),
            basis=FindingBasis.STORY_HISTORY,
            investigator_visible=True,
            claim_type=ClaimType.CONTRADICTION,
            verification_status=ClaimStatus.CONTRADICTED,
            strategic_value="high",
            evidentiary_impact=contradiction.evidentiary_impact,
            risk_profile=RiskProfile(story_contradiction=True),
        ))

    # An account where nothing can be checked is a finding in its own right,
    # scored the same way a self-contradiction is. Without it, a suspect who
    # answers every question with something unverifiable ("can't recall",
    # "it was standard", "it's on the statement") banks a flat few points per
    # turn and the pattern itself — the thing an investigator would actually
    # find damning — counts for nothing. Flagged once by the narrative agent,
    # so this scores once, not per turn.
    if narrative.unfalsifiable_account and not state.unfalsifiable_flagged:
        state.unfalsifiable_flagged = True
        pattern = narrative.unfalsifiable_account
        results.append(CheckResult(
            quoted_evidence=pattern.claim_text,
            rationale=(
                f"Nothing in the account can be checked. e.g. \"{pattern.earlier_statement}\" "
                f"and \"{pattern.later_statement}\""
            ),
            basis=FindingBasis.STORY_HISTORY,
            investigator_visible=True,
            claim_type=ClaimType.CONTRADICTION,
            verification_status=ClaimStatus.CONTRADICTED,
            strategic_value="high",
            evidentiary_impact=pattern.evidentiary_impact,
            risk_profile=RiskProfile(credibility_issue=True),
        ))

    results = dedupe_results(results)

    turn_delta, state = process_turn_scoring(results, state, player_answer)

    for result in results:
        rp = result.risk_profile
        print(
            "  [checker] "
            f"basis={result.basis.value} | visible={result.investigator_visible} | "
            f"status={result.verification_status.value} | ambiguous={result.ambiguous} | "
            f"impact={result.evidentiary_impact.value} | "
            f"future={result.future_verification_value.value} | "
            f"contradiction={rp.fact_contradiction or rp.story_contradiction} | "
            f"policy_breach={rp.policy_breach} | "
            f"evasion={rp.evasion} | credibility_issue={rp.credibility_issue} — "
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

    # Strategy uses the case log plus score/budget as urgency context.
    move, war_room = run_strategist(state, return_debug=True)

    _append_debug_log(
        f"TURN {state.question_count}\n"
        f"{'-' * 60}\n"
        f"INVESTIGATOR QUESTION:\n{question}\n\n"
        f"PLAYER ANSWER (RAW):\n{player_answer}\n\n"
        f"REALITY GATE:\n{json.dumps(gate.model_dump(), indent=2, ensure_ascii=False)}\n\n"
        f"ANALYZED ANSWER:\n{analysis_answer}\n\n"
        f"SUSPECT NARRATIVE:\n{narrative.summary}\n"
        f"SELF-CONTRADICTIONS FOUND THIS TURN:\n"
        + ("\n".join(
            f"- {c.claim_text} [impact={c.evidentiary_impact.value}]\n"
            f"  earlier: \"{c.earlier_statement}\"\n  later: \"{c.later_statement}\""
            for c in narrative.self_contradictions
          ) or "- None")
        + f"\nSTALE THREAD: {narrative.stale_thread or 'None'}\n\n"
        f"EXTRACTED CLAIMS:\n"
        + _format_extracted_claims(extracted.claims)
        + f"\n\nNOVEL CLAIMS SENT TO CHECKER ({len(checkable_claims)}):\n"
        + ("\n".join(f"- {c}" for c in checkable_claims) or "- None")
        + "\n\nSKIPPED AS REITERATED:\n"
        + ("\n".join(
            f"- \"{item.text}\""
            for item in extracted.claims
            if item.status == ClaimNoveltyStatus.REITERATED
          ) or "- None")
        + "\n\nNEW DEFENSES (non-checkable narrative):\n"
        + ("\n".join(f"- {d}" for d in extracted.new_defenses) or "- None")
        + "\n\nCHECKER FINDINGS:\n"
        + _format_checker_results(results)
        + "\n\nBAD COP / SKEPTIC:\n"
        + json.dumps(war_room.skeptic.model_dump(), indent=2, ensure_ascii=False)
        + "\n\nGOOD COP / ALTERNATIVE HYPOTHESIS:\n"
        + json.dumps(war_room.alternative.model_dump(), indent=2, ensure_ascii=False)
        + "\n\nLEAD STRATEGIST:\n"
        + json.dumps(move.model_dump(), indent=2, ensure_ascii=False)
        + "\n\nSTATE AFTER TURN:\n"
        + f"score={state.score}\n"
        + f"questions={state.question_count}/{state.max_questions}\n"
        + "case_log="
        + json.dumps(state.case_log.investigator_view(), indent=2, ensure_ascii=False)
    )

    state.last_move = move.target
    state.move_history.append(state.last_move)

    print(
        f"  [strategy] target={move.target} | "
        f"remaining={state.max_questions - state.question_count}"
    )

    line, speaker_grounding = run_speaker(
        state.case,
        move,
        state.transcript,
        score=state.score,
        case_log=state.case_log,
        remaining=max(state.max_questions - state.question_count, 0),
        last_turn_usefulness=state.last_turn_usefulness,
        narrative_summary=narrative.summary,
        return_debug=True,
    )
    state.grounding_history.append(list(speaker_grounding))
    state.transcript.append({"role": "investigator", "text": line})

    _append_debug_log(
        f"SPEAKER OUTPUT AFTER TURN {state.question_count}:\n{line}\n\n"
        # Kept in the log so fabrications stay auditable after the fact: every
        # specific claim the line makes should appear here with a real source.
        # Anything asserted in the line but missing from this list is the
        # Speaker inventing something.
        f"SPEAKER GROUNDING (claim -> source):\n"
        + ("\n".join(f"- {g}" for g in speaker_grounding)
           or "- None (line asserts nothing specific)")
        + "\n" + "=" * 60
    )

    return state, line


def get_final_resolution(state: GameState) -> str:
    return format_resolution(run_resolution(state))


if __name__ == "__main__":
    reset_debug_log()
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