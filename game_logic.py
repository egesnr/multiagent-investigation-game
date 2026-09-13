"""
Deterministic reducer for interview state and scoring.

Score means current investigator-visible incriminating evidentiary strength.
It does not reward mere talking, unsupported stories, or hidden truth.
"""

from models import (
    CheckResult,
    ClaimRecord,
    ClaimStatus,
    EvidentiaryImpact,
    FindingBasis,
    FutureVerificationValue,
    GameState,
    InvestigationLead,
    LeadStatus,
)


TIER_TABLE = [
    (0, 20, 1),
    (20, 40, 2),
    (40, 60, 3),
    (60, 80, 4),
    (80, 10**9, 5),
]

IMPACT_POINTS = {
    EvidentiaryImpact.NONE: 0,
    EvidentiaryImpact.WEAK: 4,
    EvidentiaryImpact.MODERATE: 8,
    EvidentiaryImpact.STRONG: 15,
    EvidentiaryImpact.DECISIVE: 25,
}

MAX_TURN_DAMAGE = 30


def apply_tier(score: int) -> int:
    for lo, hi, tier in TIER_TABLE:
        if lo <= score < hi:
            return tier
    return 5


def _claim_key(result: CheckResult) -> str:
    """Stable-ish key used only to stop the same finding being farmed repeatedly."""
    if result.fact_id:
        return f"{result.basis.value}:fact:{result.fact_id}:{result.verification_status.value}"
    return (
        f"{result.basis.value}:{result.verification_status.value}:"
        f"{result.quoted_evidence.strip().lower()}"
    )


def _lead_key(text: str) -> str:
    return " ".join(text.lower().split())


def _effective_status(result: CheckResult) -> ClaimStatus:
    """
    Downgrade an ambiguous extraction's status before it is treated as a firm
    finding. The Extractor flags `ambiguous=True` when it could not cleanly
    resolve who/what a claim referred to and had to fall back to a literal
    reading. A claim built on an unresolved reading must never be allowed to
    stand as a "contradicted" story-history finding, since the apparent
    conflict may be an artifact of the extraction, not something the suspect
    actually said. Contradictions against hard case facts (investigator
    evidence/policy) are unaffected, since those don't depend on comparing two
    of the suspect's own ambiguous statements against each other.
    """
    if (
        result.ambiguous
        and result.basis == FindingBasis.STORY_HISTORY
        and result.verification_status == ClaimStatus.CONTRADICTED
    ):
        return ClaimStatus.UNVERIFIED
    return result.verification_status


def update_case_log(results: list[CheckResult], state: GameState) -> GameState:
    existing_claims = {
        (c.text.strip().lower(), c.related_fact_id, c.status.value, c.basis.value)
        for c in state.case_log.claims
    }
    existing_leads = {_lead_key(lead.topic): lead for lead in state.case_log.leads}

    for result in results:
        text = result.quoted_evidence.strip()
        status = _effective_status(result)
        signature = (
            text.lower(),
            result.fact_id,
            status.value,
            result.basis.value,
        )

        if signature not in existing_claims:
            state.case_log.claims.append(
                ClaimRecord(
                    text=text,
                    turn=state.question_count,
                    status=status,
                    related_fact_id=result.fact_id,
                    rationale=result.rationale,
                    basis=result.basis,
                    investigator_visible=result.investigator_visible,
                    relation=result.relation,
                    claim_type=result.claim_type,
                    strategic_value=result.strategic_value,
                    future_verification_value=result.future_verification_value,
                    evidentiary_impact=result.evidentiary_impact,
                    suggested_thread=result.suggested_thread,
                    ambiguous=result.ambiguous,
                )
            )
            existing_claims.add(signature)

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

        if result.risk_profile.evasion:
            if text not in state.case_log.evasions:
                state.case_log.evasions.append(text)

        # A concrete, materially useful claim that needs outside checking is already
        # valuable interview work. Park it rather than asking the same thing again.
        if result.suggested_thread:
            key = _lead_key(result.suggested_thread)
            lead = existing_leads.get(key)
            if lead is None:
                lead = InvestigationLead(
                    topic=result.suggested_thread,
                    source_claim=text,
                    status=LeadStatus.OPEN,
                    importance=result.strategic_value,
                    future_verification_value=result.future_verification_value,
                )
                state.case_log.leads.append(lead)
                existing_leads[key] = lead

            if (
                status == ClaimStatus.UNVERIFIED
                and result.future_verification_value == FutureVerificationValue.HIGH
            ):
                lead.status = LeadStatus.PENDING_VERIFICATION
                lead.notes = "Material commitment captured; requires outside verification."
                if lead.topic not in state.case_log.parked_threads:
                    state.case_log.parked_threads.append(lead.topic)

    return state


def calculate_claim_score(result: CheckResult, state: GameState) -> tuple[int, str | None]:
    """
    Convert the Checker's semantic evidentiary impact into deterministic points.

    Unverified claims get zero arrest points even if they are strategically valuable.
    They remain useful through future_verification_value and the case log.
    """
    if not result.investigator_visible:
        return 0, None

    status = _effective_status(result)

    if status == ClaimStatus.UNVERIFIED:
        return 0, None

    points = IMPACT_POINTS[result.evidentiary_impact]
    if points <= 0:
        return 0, None

    key = _claim_key(result)
    if key in state.scored_findings:
        return 0, None

    state.scored_findings.append(key)
    return points, f"{result.evidentiary_impact.value} established finding (+{points})"


def _turn_usefulness(results: list[CheckResult], delta: int) -> str:
    if delta > 0:
        return "new_evidence"
    if any(r.future_verification_value == FutureVerificationValue.HIGH for r in results):
        return "high_future_value"
    if any(r.risk_profile.evasion for r in results):
        return "evasion_no_gain"
    if any(r.strategic_value in {"high", "medium"} for r in results):
        return "material_but_unresolved"
    if results:
        return "low_value"
    return "no_useful_answer"


def process_turn_scoring(
    results: list[CheckResult],
    state: GameState,
    player_answer: str = "",
) -> tuple[int, GameState]:
    raw_delta = 0
    breakdown: list[str] = []

    for result in results:
        pts, reason = calculate_claim_score(result, state)
        raw_delta += pts
        if reason:
            breakdown.append(f"'{result.quoted_evidence}' -> {reason}")

    state = update_case_log(results, state)

    final_delta = min(raw_delta, MAX_TURN_DAMAGE)
    state.score += final_delta
    state.last_turn_delta = final_delta
    state.last_turn_usefulness = _turn_usefulness(results, final_delta)
    state.tier = apply_tier(state.score)

    print("\n--- SCORE BREAKDOWN ---")
    if breakdown:
        for item in breakdown:
            print(f"  • {item}")
    else:
        print("  • No new established incriminating evidence (+0)")

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
    # Do not end immediately at the arrest threshold; the interview still has value.
    # Only an overwhelmingly strong case can end early, and never before three answers.
    return state.score >= state.case.arrest_threshold + 30 and state.question_count >= 3