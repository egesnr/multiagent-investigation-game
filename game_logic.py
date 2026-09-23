"""
Deterministic reducer for interview state and scoring.

Score means current investigator-visible incriminating evidentiary strength.
It does not reward mere talking, unsupported stories, or hidden truth.
"""

from models import (
    CheckResult,
    ClaimRecord,
    ClaimStatus,
    ClaimType,
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

# Refusing to engage is charged per DISTINCT subject refused, escalating, and
# it does not come back off. Measured against the old rule this replaces: eight
# turns of flat denial cost 24 and finished one point under the threshold,
# because the charge fired only every third turn — refusing for eight turns
# cost the same per turn as refusing for three.
#
# Escalating on distinct subjects rather than on repetitions is deliberate. It
# means saying "no comment" five times about the receipt is one refusal, not
# five, and the only way the charge grows is if the suspect refuses to engage
# with the case broadly — which is the thing an investigator would actually
# hold against them.
STALL_CHARGE = [6, 12, 18, 24]
STALL_CHARGE_MAX = 30

# The other half of a dodge, and the reversible one: the investigator asked the
# suspect to produce or account for something, and was refused, so the record
# takes it as not existing. Returned in full if they later answer it — the fact
# is released, the stall charge above is not.
DODGE_FACT_POINTS = 8

# An unverified defense the suspect cannot substantiate raises suspicion even
# though nothing is proven yet. Scaled by how central the Checker judged the
# claim: an unsupported excuse about the core allegation counts, idle detail
# does not. Refunded by resolution if the claim is later confirmed true.
PROVISIONAL_POINTS = {"high": 8, "medium": 5, "low": 0}


def apply_tier(score: int) -> int:
    for lo, hi, tier in TIER_TABLE:
        if lo <= score < hi:
            return tier
    return 5


def _claim_key(result: CheckResult) -> str:
    """Identity of a finding, used to stop the same one being farmed repeatedly.

    Keyed on the claim's SUBJECT, not its wording. Keying on prose was the
    whole reason a suspect could be scored six times for one behaviour: six
    insults are six different sentences, so they were six different keys and
    six separate strong findings, +90 for what an investigator would record
    once as "refused to cooperate and was hostile throughout". Any restatement
    dressed in new words defeated a text key by construction.

    verification_status stays in the key deliberately: a subject that was an
    unverified excuse earlier and is a proven contradiction now is genuinely
    new ground, and must still be able to score.
    """
    if result.fact_id:
        return f"{result.basis.value}:fact:{result.fact_id}:{result.verification_status.value}"

    subject = result.subject.strip().lower()
    if not subject:
        # Nothing should reach here. A subjectless finding falls back to a
        # prose key, and a prose key is the bug this whole function exists to
        # kill — it is what let one contradiction score seven times under
        # seven different sentences. Every producer of a CheckResult must
        # supply a subject; say so loudly rather than silently mis-scoring.
        print(
            "  [warning] finding has no subject, scoring falls back to prose: "
            f"{result.quoted_evidence[:70]}"
        )
        return (
            f"{result.basis.value}:{result.verification_status.value}:"
            f"{result.quoted_evidence.strip().lower()}"
        )

    # Two bands only. Which shade of established a finding is — supported,
    # admitted, contradicted — and which stage of the Checker produced it are
    # not facts about the suspect; they are routing. Keeping them in the key
    # let one behaviour score again every time it was classified differently.
    band = (
        "provisional"
        if result.verification_status == ClaimStatus.UNVERIFIED
        else "established"
    )
    return f"{band}:{subject}"


def charge_for_dodge(state: GameState, subject: str, settles_fact: bool) -> tuple[int, list[str]]:
    """Charge a refusal to engage with `subject`, and say what it cost.

    Two different things happen and they behave differently afterwards:
      - the stall charge, which is about conduct and is permanent
      - the fact charge, which stands in for evidence the suspect would not
        produce, and is refunded the moment they do
    """
    subject = subject.strip()
    if not subject:
        return 0, []

    lines: list[str] = []
    points = 0

    first_time = subject not in state.dodged_subjects
    if first_time:
        state.dodged_subjects.append(subject)
        index = len(state.dodged_subjects) - 1
        charge = (
            STALL_CHARGE[index] if index < len(STALL_CHARGE) else STALL_CHARGE_MAX
        )
        points += charge
        lines.append(
            f"refused to engage with '{subject}' "
            f"({_ordinal(len(state.dodged_subjects))} subject refused) (+{charge})"
        )

        # Only a question that asked for something producible settles a fact.
        # A refusal to confess establishes nothing — see the Extractor's
        # dodge_settles_fact rule.
        if settles_fact:
            state.dodge_fact_points[subject] = DODGE_FACT_POINTS
            points += DODGE_FACT_POINTS
            lines.append(
                f"'{subject}' taken as established, unaddressed "
                f"(+{DODGE_FACT_POINTS}, released if answered later)"
            )

    state.dodge_counts[subject] = state.dodge_counts.get(subject, 0) + 1
    return points, lines


def release_dodged_fact(state: GameState, subject: str) -> tuple[int, str | None]:
    """The suspect finally addressed something they had refused. Give back the
    fact points; keep the stall charge. Answering late does not undo having
    stalled, but it must always be worth doing."""
    subject = subject.strip()
    points = state.dodge_fact_points.pop(subject, 0)
    if not points:
        return 0, None
    return -points, f"'{subject}' answered after refusing (-{points})"


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _bank_subject(state: GameState, result: CheckResult) -> None:
    """Record, in plain words, that this subject has now been paid for."""
    subject = result.subject.strip()
    if subject and subject not in state.banked_subjects:
        state.banked_subjects.append(subject)


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
    # Same identity rule as _claim_key: a claim is placed by its subject, so a
    # reworded restatement lands on the entry already there instead of adding a
    # near-duplicate line the War Room then reads as further evidence.
    existing_claims = {
        (
            (c.subject.strip().lower() or c.text.strip().lower()),
            c.related_fact_id,
            c.status.value,
            c.basis.value,
        )
        for c in state.case_log.claims
    }
    existing_leads = {_lead_key(lead.topic): lead for lead in state.case_log.leads}

    for result in results:
        text = result.quoted_evidence.strip()
        status = _effective_status(result)
        signature = (
            result.subject.strip().lower() or text.lower(),
            result.fact_id,
            status.value,
            result.basis.value,
        )

        if signature not in existing_claims:
            state.case_log.claims.append(
                ClaimRecord(
                    subject=result.subject,
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

    An unverified claim scores no established points, but an unverified
    *defense* scores provisional suspicion — see PROVISIONAL_POINTS. Testing
    showed a careful liar was strictly better off than an honest suspect:
    every excuse he could not be caught on scored exactly zero, so six
    fabrications left him on 16 points while a man who admitted everything
    sat on 72. Suspicion now goes on the board when an excuse cannot be
    backed up, and resolution refunds it if the claim later checks out true.
    """
    if not result.investigator_visible:
        return 0, None

    status = _effective_status(result)

    if status == ClaimStatus.UNVERIFIED:
        if result.claim_type != ClaimType.DEFENSE:
            return 0, None
        points = PROVISIONAL_POINTS.get(result.strategic_value, 0)
        if points <= 0:
            return 0, None
        key = _claim_key(result)
        if key in state.scored_findings:
            return 0, None
        state.scored_findings.append(key)
        _bank_subject(state, result)
        # Recorded per-claim so resolution can take it back off the board if
        # the claim turns out to be true.
        state.provisional_findings[result.quoted_evidence.strip()] = points
        return points, f"unverified defense, provisional suspicion (+{points})"

    points = IMPACT_POINTS[result.evidentiary_impact]
    if points <= 0:
        return 0, None

    key = _claim_key(result)
    if key in state.scored_findings:
        return 0, None

    state.scored_findings.append(key)
    _bank_subject(state, result)
    return points, f"{result.evidentiary_impact.value} established finding (+{points})"


def _turn_usefulness(results: list[CheckResult], delta: int) -> str:
    # Only points from an *established* finding count as new evidence. A turn
    # whose whole gain was provisional suspicion on an unsupported excuse has
    # not proven anything, and telling the Strategist otherwise would have it
    # believe a move landed hard evidence when it merely noted a story nobody
    # can check yet.
    established = any(
        _effective_status(r) != ClaimStatus.UNVERIFIED
        and IMPACT_POINTS[r.evidentiary_impact] > 0
        for r in results
    )
    if established and delta > 0:
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
    dodged_subject: str | None = None,
    dodge_settles_fact: bool = False,
    answered_subjects: list[str] | None = None,
) -> tuple[int, GameState]:
    raw_delta = 0
    breakdown: list[str] = []

    for result in results:
        pts, reason = calculate_claim_score(result, state)
        raw_delta += pts
        if reason:
            breakdown.append(f"'{result.quoted_evidence}' -> {reason}")

    state = update_case_log(results, state)

    # No per-turn cap: a cap discarded legitimately earned points purely based
    # on which turn they happened to land in, so the same suspect making the
    # same admissions scored differently depending on whether their story fell
    # apart all at once or piece by piece.
    final_delta = raw_delta
    usefulness = _turn_usefulness(results, final_delta)

    # A refusal to engage is now recognised directly by the Extractor, against
    # the question that was actually asked, instead of being inferred from a
    # turn that happened to score nothing. Those are not the same thing: an
    # honest answer can score nothing, and a fluent evasion can score a little.
    if dodged_subject:
        state.consecutive_stonewall += 1
        dodge_points, dodge_lines = charge_for_dodge(
            state, dodged_subject, dodge_settles_fact
        )
        final_delta += dodge_points
        breakdown.extend(dodge_lines)

        # Surfaced in the shared case log, not only in the number, so the room
        # and the resolution can refer to the pattern rather than just feel it.
        if state.dodge_counts.get(dodged_subject.strip(), 0) >= 2:
            note = (
                f"Suspect refused twice to address {dodged_subject.strip()} "
                "after being told what the silence would be recorded as "
                "(Duty to Cooperate)."
            )
            if note not in state.case_log.credibility_flags:
                state.case_log.credibility_flags.append(note)
            # Closed to further asking. The suspect can still reopen it by
            # speaking to it; the investigator cannot re-ask it cold.
            if dodged_subject.strip() not in state.case_log.exhausted_targets:
                state.case_log.exhausted_targets.append(dodged_subject.strip())
    else:
        state.consecutive_stonewall = 0

    # Answering something previously refused releases the fact, never the
    # stall. Talking must always improve your position; it just cannot undo
    # having stalled.
    for subject in answered_subjects or []:
        refund, line = release_dodged_fact(state, subject)
        if refund:
            final_delta += refund
            breakdown.append(line)

    state.score += final_delta
    state.last_turn_delta = final_delta
    state.last_turn_usefulness = usefulness
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