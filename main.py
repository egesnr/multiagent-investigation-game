"""
Deterministic scoring and shared-context reducer.

Important design rule:
hidden world truth is NOT automatically investigator knowledge.
A contradiction that exists only against hidden truth is remembered internally
for end-of-game resolution but does not increase interview suspicion by itself.
"""

from models import (
    CheckResult,
    ClaimRecord,
    ClaimStatus,
    FindingBasis,
    GameState,
)


TIER_TABLE = [
    (0, 20, 1, ["open_question"]),
    (20, 40, 2, ["open_question", "press_inconsistency"]),
    (40, 60, 3, ["open_question", "press_inconsistency", "demand_proof"]),
    (60, 80, 4, ["press_inconsistency", "demand_proof", "accuse"]),
    (80, 10**9, 5, ["demand_proof", "accuse"]),
]

MAX_TURN_DAMAGE = 20


def apply_tier(score: int) -> tuple[int, list[str]]:
    for lo, hi, tier, moves in TIER_TABLE:
        if lo <= score < hi:
            return tier, moves
    return 5, TIER_TABLE[-1][3]


def _claim_key(result: CheckResult) -> str:
    basis = result.basis.value
    if result.fact_id:
        return f"{basis}:fact:{result.fact_id}"
    return f"{basis}:claim:{result.quoted_evidence.strip().lower()}"


def _claim_status(result: CheckResult) -> ClaimStatus:
    rp = result.risk_profile
    if rp.fact_contradiction or rp.story_contradiction:
        return ClaimStatus.CONTRADICTED
    if rp.policy_breach and not rp.proof_deficit:
        return ClaimStatus.ADMITTED
    if rp.proof_deficit:
        return ClaimStatus.UNVERIFIED
    return ClaimStatus.SUPPORTED


def update_case_log(results: list[CheckResult], state: GameState) -> GameState:
    existing = {
        (c.text.strip().lower(), c.related_fact_id, c.status.value, c.basis.value)
        for c in state.case_log.claims
    }

    for result in results:
        text = result.quoted_evidence.strip()
        status = _claim_status(result)
        signature = (
            text.lower(),
            result.fact_id,
            status.value,
            result.basis.value,
        )

        if signature not in existing:
            state.case_log.claims.append(
                ClaimRecord(
                    text=text,
                    turn=state.question_count,
                    status=status,
                    related_fact_id=result.fact_id,
                    rationale=result.rationale,
                    basis=result.basis,
                    investigator_visible=result.investigator_visible,
                    materiality=result.materiality,
                )
            )
            existing.add(signature)

        if result.basis == FindingBasis.HIDDEN_TRUTH and (
            result.risk_profile.fact_contradiction
        ):
            if text not in state.case_log.hidden_conflicts:
                state.case_log.hidden_conflicts.append(text)
            continue

        if not result.investigator_visible:
            continue

        if status == ClaimStatus.CONTRADICTED:
            if text not in state.case_log.contradictions:
                state.case_log.contradictions.append(text)

        if status == ClaimStatus.ADMITTED:
            if text not in state.case_log.admissions:
                state.case_log.admissions.append(text)

        if status == ClaimStatus.UNVERIFIED:
            if text not in state.case_log.unresolved_claims:
                state.case_log.unresolved_claims.append(text)

        if result.risk_profile.credibility_issue:
            if text not in state.case_log.credibility_flags:
                state.case_log.credibility_flags.append(text)

    return state


def calculate_claim_score(result: CheckResult, state: GameState) -> tuple[int, list[str]]:
    """
    Interview score reflects what the investigator can legitimately infer now.
    Hidden-truth-only contradictions are not scored here.
    """
    if not result.investigator_visible or result.basis == FindingBasis.HIDDEN_TRUTH:
        return 0, []

    rp = result.risk_profile
    key = _claim_key(result)
    points = 0
    reasons: list[str] = []

    if rp.fact_contradiction or rp.story_contradiction:
        if key in state.flagged_facts:
            points += 2
            reasons.append("Repeated contradiction (+2)")
        else:
            points += 18
            state.flagged_facts.append(key)
            reasons.append("Established contradiction (+18)")

    if rp.policy_breach and not rp.proof_deficit:
        policy_key = f"policy:{result.quoted_evidence.strip().lower()}"
        if policy_key not in state.flagged_facts:
            state.flagged_facts.append(policy_key)
            points += 8
            reasons.append("Established policy breach (+8)")

    if rp.proof_deficit and not (rp.fact_contradiction or rp.story_contradiction):
        text = result.quoted_evidence.strip()
        if text not in state.case_log.unresolved_claims:
            state.audit_debt += 1
            points += 3
            reasons.append("Material unverified claim (+3, audit debt +1)")

    if rp.credibility_issue and not (
        rp.fact_contradiction or rp.story_contradiction
    ):
        points += 2
        reasons.append("Material credibility concern (+2)")

    return points, reasons


def process_turn_scoring(
    results: list[CheckResult],
    state: GameState,
    player_answer: str = "",
) -> tuple[int, GameState]:
    raw_delta = 0
    breakdown: list[str] = []

    for result in results:
        pts, reasons = calculate_claim_score(result, state)
        raw_delta += pts
        if reasons:
            breakdown.append(
                f"'{result.quoted_evidence}' -> {', '.join(reasons)}"
            )

    state = update_case_log(results, state)

    final_delta = min(raw_delta, MAX_TURN_DAMAGE)
    state.score += final_delta
    state.tier, _ = apply_tier(state.score)

    print("\n--- SCORE BREAKDOWN ---")
    if breakdown:
        for item in breakdown:
            print(f"  • {item}")
    else:
        print("  • No investigator-visible risk signals (+0)")

    print(
        f"  Total Turn Delta: +{final_delta} | New Score: {state.score}"
        "\n-----------------------"
    )
    return final_delta, state


def dedupe_results(results: list[CheckResult]) -> list[CheckResult]:
    seen: set[str] = set()
    output: list[CheckResult] = []

    for result in results:
        key = _claim_key(result)
        if key in seen:
            continue
        seen.add(key)
        output.append(result)

    return output


def case_decisively_resolved(state: GameState) -> bool:
    return (
        state.score > state.case.arrest_threshold * 1.3
        and state.question_count >= 3
    )
