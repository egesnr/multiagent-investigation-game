"""
No-API smoke test for schemas and deterministic reducer.
"""

import json
from pathlib import Path

from models import (
    CaseFile,
    CheckResult,
    FindingBasis,
    GameState,
    RiskProfile,
)
from game_logic import process_turn_scoring


case_path = Path("case_files/restaurant_case.json")
case = CaseFile(**json.loads(case_path.read_text(encoding="utf-8")))
state = GameState(case=case)

# Investigator-visible story contradiction should score.
visible = CheckResult(
    quoted_evidence="I was with a colleague",
    rationale="Conflicts with an already established statement/evidence.",
    basis=FindingBasis.STORY_HISTORY,
    investigator_visible=True,
    risk_profile=RiskProfile(story_contradiction=True),
)
delta, state = process_turn_scoring([visible], state)
assert delta > 0
assert state.case_log.contradictions

# Hidden-truth-only conflict should NOT score or leak into investigator view.
hidden = CheckResult(
    quoted_evidence="A hidden-world claim",
    rationale="Conflicts only with authored truth.",
    basis=FindingBasis.HIDDEN_TRUTH,
    investigator_visible=False,
    risk_profile=RiskProfile(fact_contradiction=True),
)
before = state.score
delta, state = process_turn_scoring([hidden], state)
assert delta == 0
assert state.score == before
assert "A hidden-world claim" in state.case_log.hidden_conflicts
assert "A hidden-world claim" not in state.case_log.investigator_view()["contradictions"]

print("SMOKE TEST PASSED")
