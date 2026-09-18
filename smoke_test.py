"""No-API smoke test for schemas and deterministic reducer."""

import json
from pathlib import Path

from models import (
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
from game_logic import process_turn_scoring
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

# Sustained stonewalling (empty/evasive turns, i.e. no extracted claims at
# all) must eventually cost points under the Duty to Cooperate policy —
# otherwise pure non-cooperation is a free pass with zero downside.
stonewall_state = GameState(case=case)
before = stonewall_state.score
for _ in range(2):
    delta, stonewall_state = process_turn_scoring([], stonewall_state)
    assert delta == 0, "no penalty before the streak threshold is reached"
delta, stonewall_state = process_turn_scoring([], stonewall_state)
assert delta == 12, f"expected the obstruction penalty on turn 3, got {delta}"
assert stonewall_state.score == before + 12
assert stonewall_state.consecutive_stonewall == 3
assert stonewall_state.case_log.credibility_flags, (
    "obstruction penalty should surface as a visible case_log flag"
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
