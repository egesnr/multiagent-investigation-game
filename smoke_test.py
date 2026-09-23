"""No-API smoke test for schemas and deterministic reducer."""

import json
from pathlib import Path

from models import (
    ClaimRecord,
    CaseFile,
    CheckResult,
    ClaimStatus,
    ClaimType,
    EvidenceAssessment,
    EvidenceRelation,
    EvidentiaryImpact,
    FindingBasis,
    FutureVerificationValue,
    GameState,
    InvestigativeSignificance,
    RiskProfile,
)
from game_logic import dedupe_results, process_turn_scoring
from agents import _merge_checker_results


case_path = Path("case_files/restaurant_case.json")
case = CaseFile(**json.loads(case_path.read_text(encoding="utf-8")))
state = GameState(case=case)
state.question_count = 1

# Investigator-visible contradiction should score.
visible = CheckResult(
    quoted_evidence="I was with a colleague",
    rationale="Conflicts with established investigator-visible evidence.",
    basis=FindingBasis.INVESTIGATOR_EVIDENCE,
    investigator_visible=True,
    verification_status=ClaimStatus.CONTRADICTED,
    claim_type=ClaimType.CONTRADICTION,
    evidentiary_impact=EvidentiaryImpact.STRONG,
    risk_profile=RiskProfile(fact_contradiction=True),
)
delta, state = process_turn_scoring([visible], state)
assert delta == 15
assert state.case_log.contradictions

# An UNVERIFIED claim must never score arrest points even if evidentiary_impact
# was (incorrectly) set to something nonzero upstream. This is the code-level
# backstop for the "unverified -> impact=none" rule.
mis_tagged_unverified = CheckResult(
    quoted_evidence="A claim nobody has actually checked",
    rationale="No evidence bears on this yet.",
    basis=FindingBasis.UNRESOLVED,
    investigator_visible=True,
    verification_status=ClaimStatus.UNVERIFIED,
    claim_type=ClaimType.BACKGROUND,
    evidentiary_impact=EvidentiaryImpact.STRONG,
)
before = state.score
delta, state = process_turn_scoring([mis_tagged_unverified], state)
assert delta == 0
assert state.score == before

# A high-value unsupported defense scores provisional suspicion and is parked for
# verification. It used to score nothing, which made a careful liar strictly better
# off than an honest suspect — every excuse that couldn't be disproven on the spot
# was free. The points come back off in resolution if the claim checks out true.
future = CheckResult(
    quoted_evidence="The restaurant added an extra zero.",
    rationale="Concrete defense that requires merchant-side verification.",
    basis=FindingBasis.UNRESOLVED,
    investigator_visible=True,
    verification_status=ClaimStatus.UNVERIFIED,
    claim_type=ClaimType.DEFENSE,
    strategic_value="high",
    future_verification_value=FutureVerificationValue.HIGH,
    evidentiary_impact=EvidentiaryImpact.NONE,
    suggested_thread="restaurant amount-entry error",
)
before = state.score
delta, state = process_turn_scoring([future], state)
assert delta == 8, f"high-value unsupported defense should raise suspicion, got {delta}"
assert state.score == before + 8
assert state.provisional_findings["The restaurant added an extra zero."] == 8, (
    "provisional points must be recorded per claim so resolution can refund them"
)
assert state.last_turn_usefulness == "high_future_value"
assert "restaurant amount-entry error" in state.case_log.parked_threads

# Background chatter that happens to be unverified must still score nothing —
# suspicion attaches to excuses the suspect cannot back up, not to small talk.
chatter = CheckResult(
    quoted_evidence="I flew in on the Tuesday.",
    rationale="Ordinary background detail, nothing bears on it.",
    basis=FindingBasis.UNRESOLVED,
    investigator_visible=True,
    verification_status=ClaimStatus.UNVERIFIED,
    claim_type=ClaimType.BACKGROUND,
    strategic_value="high",
)
before = state.score
delta, state = process_turn_scoring([chatter], state)
assert delta == 0, f"unverified background must not score, got {delta}"
assert state.score == before

# One behaviour repeated in six different sentences is ONE finding. Before
# findings were keyed on subject, each rewording produced its own key and its
# own +15, so a suspect who insulted the investigator six times collected 90
# points for a single fact about the interview.
abuse_state = GameState(case=case)
abuse_wordings = [
    "The suspect directed personal abuse at the investigator.",
    "The suspect called the investigator an incompetent idiot.",
    "The suspect called the investigator a disgrace.",
    "The suspect stated the investigator is nothing.",
    "The suspect said they do not answer to the investigator.",
    "The suspect stated they are not interested in the compliance process.",
]
abuse_total = 0
for wording in abuse_wordings:
    finding = CheckResult(
        quoted_evidence=wording,
        subject="the suspect's conduct toward the investigator",
        rationale="Abuse directed at the investigator, visible in the transcript.",
        basis=FindingBasis.POLICY,
        investigator_visible=True,
        verification_status=ClaimStatus.SUPPORTED,
        claim_type=ClaimType.OTHER,
        strategic_value="high",
        evidentiary_impact=EvidentiaryImpact.STRONG,
    )
    delta, abuse_state = process_turn_scoring([finding], abuse_state)
    abuse_total += delta
assert abuse_total == 15, f"one conduct subject must score once, got {abuse_total}"
assert len(abuse_state.case_log.claims) == 1, (
    f"one subject must hold one case-log entry, got {len(abuse_state.case_log.claims)}"
)

# ...and re-badging that same conduct as ADMITTED, or routing it through a
# different Checker stage, is not a second fact about the suspect. Only the
# established/provisional band counts, so this must not score again.
rebadged = CheckResult(
    quoted_evidence="The suspect stated they do not answer to the investigator.",
    subject="the suspect's conduct toward the investigator",
    rationale="Same conduct, classified as an admission this time.",
    basis=FindingBasis.STORY_HISTORY,
    investigator_visible=True,
    verification_status=ClaimStatus.ADMITTED,
    claim_type=ClaimType.ADMISSION,
    strategic_value="high",
    evidentiary_impact=EvidentiaryImpact.STRONG,
)
delta, abuse_state = process_turn_scoring([rebadged], abuse_state)
assert delta == 0, f"re-badged conduct must not score again, got {delta}"

# The band that does matter: an excuse nobody could check earns provisional
# suspicion, and later proving it false is genuinely new ground.
band_state = GameState(case=case)
unchecked = CheckResult(
    quoted_evidence="The dinner was with a client.",
    subject="who the suspect dined with",
    rationale="No evidence either way yet.",
    basis=FindingBasis.UNRESOLVED,
    investigator_visible=True,
    verification_status=ClaimStatus.UNVERIFIED,
    claim_type=ClaimType.DEFENSE,
    strategic_value="high",
)
delta, band_state = process_turn_scoring([unchecked], band_state)
assert delta == 8, f"unverified defense should score provisionally, got {delta}"
disproved = CheckResult(
    quoted_evidence="The suspect dined alone.",
    subject="who the suspect dined with",
    rationale="Booking shows a table for one.",
    basis=FindingBasis.INVESTIGATOR_EVIDENCE,
    investigator_visible=True,
    verification_status=ClaimStatus.CONTRADICTED,
    claim_type=ClaimType.CONTRADICTION,
    strategic_value="high",
    evidentiary_impact=EvidentiaryImpact.STRONG,
)
delta, band_state = process_turn_scoring([disproved], band_state)
assert delta == 15, f"proving the excuse false is new ground, got {delta}"

# Within a single turn, restatements of one subject collapse before scoring.
turn_state = GameState(case=case)
same_turn = [
    CheckResult(
        quoted_evidence=w,
        subject="whether any wrongdoing occurred",
        rationale="Blanket denial.",
        basis=FindingBasis.POLICY,
        investigator_visible=True,
        verification_status=ClaimStatus.SUPPORTED,
        claim_type=ClaimType.OTHER,
        strategic_value="high",
        evidentiary_impact=EvidentiaryImpact.MODERATE,
    )
    for w in ("I did nothing wrong.", "I dispute the premise entirely.")
]
delta, turn_state = process_turn_scoring(dedupe_results(same_turn), turn_state)
assert delta == 8, f"one subject in one turn scores once, got {delta}"

# Self-contradictions are built in code rather than coming from the Extractor,
# and for a while they were the one finding type with no subject — so each
# rewording of one tension scored again. Measured live at seven scorings of a
# single contradiction, 84 points of a 132-point run.
story_state = GameState(case=case)
rewordings = [
    "The suspect claims to know the bill was $320 yet never saw the final bill.",
    "The suspect asserts a precise $320 figure but cannot name a single dish.",
    "The suspect states the amount was $320 yet cannot remember the restaurant.",
]
story_total = 0
for wording in rewordings:
    story_total_before = story_total
    finding = CheckResult(
        quoted_evidence=wording,
        subject="the suspect's basis for the $320 figure",
        rationale="One tension, restated.",
        basis=FindingBasis.STORY_HISTORY,
        investigator_visible=True,
        verification_status=ClaimStatus.CONTRADICTED,
        claim_type=ClaimType.CONTRADICTION,
        strategic_value="high",
        evidentiary_impact=EvidentiaryImpact.STRONG,
    )
    delta, story_state = process_turn_scoring([finding], story_state)
    story_total += delta
assert story_total == 15, f"one tension must score once, got {story_total}"

# A degenerate model output must not be able to poison the shared state.
# Observed live: 109,259 characters of keyboard mash in suggested_thread,
# which then rode along in the case log to every agent on every later turn
# and turned a ninety-second turn into thirteen minutes of rate-limit retries.
runaway = ClaimRecord(text="x" * 109259, turn=1, suggested_thread="y" * 50000)
assert len(runaway.text) < 4100, f"runaway text not truncated: {len(runaway.text)}"
assert len(runaway.suggested_thread) < 4100
assert "truncated" in runaway.text
assert ClaimRecord(text="a normal claim", turn=1).text == "a normal claim", (
    "ordinary text must pass through untouched"
)

# Refusing to engage is charged per DISTINCT subject refused, escalating, so
# refusing broadly costs far more than refusing the same thing repeatedly.
dodge_state = GameState(case=case)

# 1st subject: stall 6 + the fact the suspect would not produce, 8.
delta, dodge_state = process_turn_scoring(
    [], dodge_state, dodged_subject="the itemized receipt", dodge_settles_fact=True
)
assert delta == 14, f"first dodge should cost 6 + 8, got {delta}"

# Refusing the SAME subject again is not a second refusal — it is the same one,
# and the only thing it does is close the subject to further asking.
delta, dodge_state = process_turn_scoring(
    [], dodge_state, dodged_subject="the itemized receipt", dodge_settles_fact=True
)
assert delta == 0, f"re-refusing one subject must not charge again, got {delta}"
assert "the itemized receipt" in dodge_state.case_log.exhausted_targets, (
    "after a second refusal the subject closes to further asking"
)
assert dodge_state.case_log.credibility_flags

# A second, different subject costs more than the first did.
delta, dodge_state = process_turn_scoring(
    [], dodge_state, dodged_subject="who else was at the table", dodge_settles_fact=True
)
assert delta == 20, f"second distinct subject should cost 12 + 8, got {delta}"

# Silence can never establish a state of mind: refusing to confess settles no
# fact, so only the stall is charged.
delta, dodge_state = process_turn_scoring(
    [], dodge_state, dodged_subject="whether they knew it was improper",
    dodge_settles_fact=False,
)
assert delta == 18, f"a refusal to confess charges the stall only, got {delta}"

# Answering something previously refused releases the fact and keeps the stall.
before = dodge_state.score
delta, dodge_state = process_turn_scoring(
    [], dodge_state, answered_subjects=["the itemized receipt"]
)
assert delta == -8, f"answering should return the fact points, got {delta}"
assert dodge_state.score == before - 8
assert "the itemized receipt" not in dodge_state.dodge_fact_points
# ...but the stall charge never comes back off.
assert "the itemized receipt" in dodge_state.dodged_subjects

# Refusing every subject put to you reaches the arrest threshold on its own,
# with no help from the evidence: 14 + 20 + 18 + 26 + 32 = 110 by the fifth.
alone = GameState(case=case)
for i, subject in enumerate(["a", "b", "c", "d", "e"]):
    _, alone = process_turn_scoring(
        [], alone, dodged_subject=subject, dodge_settles_fact=True
    )
assert alone.score >= case.arrest_threshold, (
    f"total refusal should reach the threshold alone, got {alone.score}"
)

# --- agents._merge_checker_results: force-derived fields never drift ---
# investigator_visible is always True (no LLM-controlled field for it exists
# on EvidenceAssessment at all anymore).
ev_contradicted = EvidenceAssessment(
    claim="The amount was $320",
    claim_proposition="The transaction was $320",
    evidence_proposition="Records show $3,200",
    relation=EvidenceRelation.CONTRADICTS,
    rationale="Direct contradiction.",
    basis=FindingBasis.INVESTIGATOR_EVIDENCE,
    is_admission=False,
)
ev_contradicted.verification_status = ClaimStatus.CONTRADICTED

# A claim the Stage 2 model mistakenly marks policy_breach=True even though
# verification_status is UNVERIFIED: the merge must force it back to False.
ev_unverified = EvidenceAssessment(
    claim="The suspect missed the 24-hour reporting window",
    claim_proposition="The suspect did not report within 24 hours",
    evidence_proposition="none",
    relation=EvidenceRelation.NEUTRAL,
    rationale="Nothing yet establishes this either way.",
    basis=FindingBasis.UNRESOLVED,
    is_admission=False,
)
ev_unverified.verification_status = ClaimStatus.UNVERIFIED

sig_contradicted = InvestigativeSignificance(
    claim=ev_contradicted.claim,
    risk_profile=RiskProfile(),
    evidentiary_impact=EvidentiaryImpact.STRONG,
)
sig_unverified = InvestigativeSignificance(
    claim=ev_unverified.claim,
    risk_profile=RiskProfile(policy_breach=True),
    evidentiary_impact=EvidentiaryImpact.NONE,
)

merged = _merge_checker_results(
    [ev_contradicted, ev_unverified],
    [sig_contradicted, sig_unverified],
)
assert all(r.investigator_visible is True for r in merged)
assert merged[0].risk_profile.fact_contradiction is True
assert merged[1].risk_profile.policy_breach is False, (
    "policy_breach must be forced False for an UNVERIFIED claim"
)

print("SMOKE TEST PASSED")
