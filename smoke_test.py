"""No-API smoke test for schemas and deterministic reducer."""

import json
from pathlib import Path

from models import (
    CaseFile,
    CheckResult,
    ClaimStatus,
    ClaimType,
    EvidentiaryImpact,
    FindingBasis,
    FutureVerificationValue,
    GameState,
    RiskProfile,
)
from game_logic import process_turn_scoring


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

# Hidden-truth-only conflict should NOT score or leak.
hidden = CheckResult(
    quoted_evidence="A hidden-world claim",
    rationale="Conflicts only with authored truth.",
    basis=FindingBasis.HIDDEN_TRUTH,
    investigator_visible=False,
    verification_status=ClaimStatus.CONTRADICTED,
    claim_type=ClaimType.CONTRADICTION,
    evidentiary_impact=EvidentiaryImpact.NONE,
    risk_profile=RiskProfile(fact_contradiction=True),
)
before = state.score
delta, state = process_turn_scoring([hidden], state)
assert delta == 0
assert state.score == before
assert "A hidden-world claim" in state.case_log.hidden_conflicts
assert "A hidden-world claim" not in state.case_log.investigator_view()["contradictions"]

# A high-value unsupported defense gets 0 arrest points but is parked for verification.
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
assert delta == 0
assert state.score == before
assert state.last_turn_usefulness == "high_future_value"
assert "restaurant amount-entry error" in state.case_log.parked_threads

print("SMOKE TEST PASSED")
