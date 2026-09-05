"""
Core schemas for the open-world multi-agent investigation game.

The authored CaseFile is fixed world truth.
GameState is the shared interview context.
Hidden truth may be used by the Checker/Resolution agent, but it must not leak
into the War Room or Speaker unless the investigator has actually established it.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Certainty(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class Weight(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Fact(BaseModel):
    id: str
    description: str
    true_value: str
    weight: Weight
    certainty: Certainty
    visible_to: list[str]


class CaseFile(BaseModel):
    scenario_type: str
    persona: str
    guilty: bool
    facts: list[Fact]
    policy_rules: list[str] = Field(default_factory=list)
    arrest_threshold: int = 80
    claim_categories: list[str] = Field(default_factory=list)

    def visible_facts(self, actor: str) -> list[Fact]:
        return [f for f in self.facts if actor in f.visible_to]


class FindingBasis(str, Enum):
    INVESTIGATOR_EVIDENCE = "investigator_evidence"
    STORY_HISTORY = "story_history"
    HIDDEN_TRUTH = "hidden_truth"
    POLICY = "policy"
    UNRESOLVED = "unresolved"
    NONE = "none"


class RiskProfile(BaseModel):
    fact_contradiction: bool = False
    story_contradiction: bool = False
    policy_breach: bool = False
    proof_deficit: bool = False
    credibility_issue: bool = False


class CheckResult(BaseModel):
    fact_id: Optional[str] = None
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)
    quoted_evidence: str
    rationale: str
    basis: FindingBasis = FindingBasis.NONE
    investigator_visible: bool = True
    materiality: str = "medium"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    ADMITTED = "admitted"


class ClaimRecord(BaseModel):
    text: str
    turn: int
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    related_fact_id: Optional[str] = None
    rationale: Optional[str] = None
    basis: FindingBasis = FindingBasis.NONE
    investigator_visible: bool = True
    materiality: str = "medium"


class LeadStatus(str, Enum):
    OPEN = "open"
    PENDING_VERIFICATION = "pending_verification"
    RESOLVED = "resolved"
    EXHAUSTED = "exhausted"


class InvestigationLead(BaseModel):
    topic: str
    source_claim: Optional[str] = None
    status: LeadStatus = LeadStatus.OPEN
    importance: str = "medium"
    notes: Optional[str] = None


class CaseLog(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)
    leads: list[InvestigationLead] = Field(default_factory=list)

    contradictions: list[str] = Field(default_factory=list)
    hidden_conflicts: list[str] = Field(default_factory=list)
    admissions: list[str] = Field(default_factory=list)
    active_defenses: list[str] = Field(default_factory=list)
    credibility_flags: list[str] = Field(default_factory=list)

    evidence_requested: list[str] = Field(default_factory=list)
    unresolved_claims: list[str] = Field(default_factory=list)
    exhausted_targets: list[str] = Field(default_factory=list)

    def investigator_view(self) -> dict:
        """Safe shared context for War Room/Speaker; hidden-truth findings are removed."""
        return {
            "claims": [
                c.model_dump()
                for c in self.claims
                if c.investigator_visible
            ],
            "leads": [lead.model_dump() for lead in self.leads],
            "contradictions": list(self.contradictions),
            "admissions": list(self.admissions),
            "active_defenses": list(self.active_defenses),
            "credibility_flags": list(self.credibility_flags),
            "evidence_requested": list(self.evidence_requested),
            "unresolved_claims": list(self.unresolved_claims),
            "exhausted_targets": list(self.exhausted_targets),
        }


class StrategistMove(BaseModel):
    target: str = Field(description="Free-form investigation thread to pursue next")
    tactic: str = Field(description="Allowed interview tactic")
    rationale: str = Field(description="Internal strategy rationale")


class VerificationStatus(str, Enum):
    CONFIRMED = "confirmed"
    DISPROVED = "disproved"
    INCONCLUSIVE = "inconclusive"
    NOT_MATERIAL = "not_material"


class VerificationResult(BaseModel):
    claim: str
    status: VerificationStatus
    basis: str
    consequence: str


class FinalOutcome(str, Enum):
    CAUGHT = "CAUGHT"
    NOT_PROVEN = "NOT_PROVEN"
    POLICY_VIOLATION_ONLY = "POLICY_VIOLATION_ONLY"


class ResolutionReport(BaseModel):
    verifications: list[VerificationResult] = Field(default_factory=list)
    outcome: FinalOutcome
    confidence: str
    reasoning: str
    aftermath: str


class GameState(BaseModel):
    case: CaseFile
    transcript: list[dict] = Field(default_factory=list)
    case_log: CaseLog = Field(default_factory=CaseLog)

    score: int = 0
    audit_debt: int = 0
    tier: int = 1
    question_count: int = 0
    max_questions: int = 7

    flagged_facts: list[str] = Field(default_factory=list)
    last_move: Optional[str] = None
    move_history: list[str] = Field(default_factory=list)
