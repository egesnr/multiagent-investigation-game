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


class SuspectDemeanor(str, Enum):
    """How the suspect is coming across in their CURRENT answer only — a
    behavioral read, not an evidentiary one. Read fresh each turn (see
    ExtractedClaims.demeanor); any trend across turns is derived in code from
    the stored history, never re-asked of an LLM."""

    COMPOSED = "composed"
    COOPERATIVE = "cooperative"
    DEFENSIVE = "defensive"
    EVASIVE = "evasive"
    AGGRESSIVE = "aggressive"
    PANICKED = "panicked"


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

    # Physical/world state is deliberately narrow. Open-world speech is allowed,
    # but the player cannot conjure objects or people into the room by typing it.
    room_objects: list[str] = Field(default_factory=list)
    present_people: list[str] = Field(default_factory=lambda: ["suspect", "investigator"])

    def visible_facts(self, actor: str) -> list[Fact]:
        return [f for f in self.facts if actor in f.visible_to]


def format_facts(facts: list[Fact]) -> str:
    """Shared fact-rendering used by every LLM-facing prompt that reasons
    over authored facts (Checker, War Room, Resolution), so weight/certainty
    are always shown consistently instead of being silently dropped in some
    call sites and not others."""
    if not facts:
        return "- None"
    return "\n".join(
        f"- {f.id}: {f.description} = {f.true_value} "
        f"[weight={f.weight.value}, certainty={f.certainty.value}]"
        for f in facts
    )


class FindingBasis(str, Enum):
    INVESTIGATOR_EVIDENCE = "investigator_evidence"
    STORY_HISTORY = "story_history"
    POLICY = "policy"
    UNRESOLVED = "unresolved"


class ClaimType(str, Enum):
    """
    ADMISSION and CONTRADICTION are always forced in code from
    verification_status (never left to the LLM to re-guess).
    DEFENSE is always forced in code from the Extractor's is_defense flag
    (this absorbs the old EXPLANATION bucket, which had no content distinct
    from DEFENSE). EVASION was dropped entirely as a claim_type: it already
    exists as risk_profile.evasion and duplicating it here just invited the
    two to disagree.
    Only BACKGROUND and OTHER are genuine free judgment calls left for the
    Stage 2 LLM.
    """
    BACKGROUND = "background"
    DEFENSE = "defense"
    ADMISSION = "admission"
    CONTRADICTION = "contradiction"
    OTHER = "other"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    ADMITTED = "admitted"


class EvidentiaryImpact(str, Enum):
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    DECISIVE = "decisive"


class FutureVerificationValue(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskProfile(BaseModel):
    fact_contradiction: bool = False
    story_contradiction: bool = False
    policy_breach: bool = False
    credibility_issue: bool = False
    evasion: bool = False


class EvidenceRelation(str, Enum):
    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"

class EvidenceAssessment(BaseModel):
    claim: str = Field(...)
    claim_proposition: str = Field(...)
    evidence_proposition: str = Field(...)
    relation: EvidenceRelation = Field(...)

    fact_id: Optional[str] = None
    rationale: str
    basis: FindingBasis = FindingBasis.UNRESOLVED
    # No investigator_visible field here: Stage 1 is only ever shown
    # investigator-visible facts (hidden authored truth never reaches the
    # interview pipeline), so every EvidenceAssessment is visible by
    # construction. CheckResult still carries investigator_visible=True as a
    # forced constant (see agents._merge_checker_results) so downstream
    # filtering code keeps a hook for that invariant, but nothing upstream is
    # ever asked to re-decide it.
    is_admission: bool = Field(
        default=False,
        description="True only if the suspect explicitly and directly confesses "
        "to a materially damaging fact in their own words, independent of any "
        "evidence comparison."
    )
    verification_status: ClaimStatus = ClaimStatus.UNVERIFIED  # computed after LLM call, not model output
class InvestigativeSignificance(BaseModel):
    """Stage 2: interpret investigative significance after status is fixed."""

    claim: str
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)
    claim_type: ClaimType = ClaimType.BACKGROUND
    strategic_value: str = Field(default="low", description="One of: low, medium, high")
    future_verification_value: FutureVerificationValue = FutureVerificationValue.NONE
    evidentiary_impact: EvidentiaryImpact = EvidentiaryImpact.NONE
    suggested_thread: Optional[str] = None


class CheckResult(BaseModel):
    """Merged downstream result. Existing reducers and logs consume this schema."""

    fact_id: Optional[str] = None
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)
    quoted_evidence: str
    rationale: str

    # Internal source/visibility boundary. This is not the claim status.
    basis: FindingBasis = FindingBasis.UNRESOLVED
    investigator_visible: bool = True

    # Preserves the Stage 1 logical comparison (ENTAILS/CONTRADICTS/NEUTRAL) that
    # verification_status was derived from, so past turns can be audited later
    # instead of only existing as a console print during the run.
    relation: EvidenceRelation = EvidenceRelation.NEUTRAL

    claim_type: ClaimType = ClaimType.BACKGROUND
    verification_status: ClaimStatus = ClaimStatus.UNVERIFIED
    strategic_value: str = Field(default="low", description="One of: low, medium, high")
    future_verification_value: FutureVerificationValue = FutureVerificationValue.NONE
    evidentiary_impact: EvidentiaryImpact = EvidentiaryImpact.NONE
    suggested_thread: Optional[str] = None

    # True when the Extractor could not cleanly resolve who/what a claim referred
    # to and had to fall back to the weakest literal reading. Ambiguous claims
    # must never be treated as a firm story-history contradiction (see game_logic).
    ambiguous: bool = False


class ClaimRecord(BaseModel):
    text: str
    turn: int
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    related_fact_id: Optional[str] = None
    rationale: Optional[str] = None
    basis: FindingBasis = FindingBasis.UNRESOLVED
    investigator_visible: bool = True
    relation: EvidenceRelation = EvidenceRelation.NEUTRAL
    claim_type: ClaimType = ClaimType.BACKGROUND
    strategic_value: str = "low"
    future_verification_value: FutureVerificationValue = FutureVerificationValue.NONE
    evidentiary_impact: EvidentiaryImpact = EvidentiaryImpact.NONE
    suggested_thread: Optional[str] = None
    ambiguous: bool = False


class ClaimNoveltyStatus(str, Enum):
    NEW = "new"
    REITERATED = "reiterated"
    UPDATED = "updated"


class ExtractedClaimItem(BaseModel):
    text: str = Field(description="The complete, standalone factual claim.")
    status: ClaimNoveltyStatus = Field(
        description="NEW if not previously stated, REITERATED if it restates a "
        "known_claim with the same meaning, UPDATED if it revises/replaces a "
        "known_claim with a materially different version."
    )
    is_defense: bool = Field(
        default=False,
        description="True if this claim functions as an excuse, alternative "
        "explanation, or unverified alibi rather than a plain factual admission.",
    )
    checkable: bool = Field(
        default=True,
        description="False only for pure opinion, emotional appeals, or rhetorical "
        "statements with no factual content to verify.",
    )
    ambiguous: bool = Field(
        default=False,
        description="True if the suspect's wording could support more than one "
        "reading (e.g. unclear who performed an action, unclear referent). When "
        "true, `text` must contain the weakest/most literal reading that avoids "
        "committing to any one interpretation.",
    )


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaimItem] = Field(default_factory=list)
    new_defenses: list[str] = Field(default_factory=list)
    demeanor: SuspectDemeanor = Field(
        default=SuspectDemeanor.COMPOSED,
        description="Read only from THIS answer's tone and behavior, not the "
        "conversation as a whole and not the factual content of the claims.",
    )
    demeanor_cue: str = Field(
        default="",
        description="One short phrase pointing at the specific tell that "
        "justifies the demeanor call (e.g. 'raised voice, challenged my "
        "authority to ask' or 'long hedge before answering, then over-"
        "explained'). Empty only if the answer is too short to show any tell.",
    )


class SelfContradiction(BaseModel):
    """A tension between two of the SUSPECT'S OWN statements, found by reading
    the whole account as one story — not a claim checked against an authored
    fact. This is the thing atomic claim-by-claim checking structurally
    cannot find on its own: it requires holding two things said in different
    turns in mind at once and noticing they don't sit together."""

    claim_text: str = Field(
        description="A single, self-contained statement of the tension for "
        "the case log and the investigator, e.g. 'The suspect's account of "
        "noticing the alert is internally inconsistent: they first denied "
        "seeing it, then said they saw it but dismissed it.'"
    )
    earlier_statement: str = Field(description="The earlier statement, quoted or closely paraphrased.")
    later_statement: str = Field(description="The later statement that sits badly against it.")
    evidentiary_impact: EvidentiaryImpact = Field(
        description="How damaging THIS specific self-contradiction is to the "
        "suspect's credibility, same rubric as evidentiary impact elsewhere: "
        "weak (minor/peripheral wording drift), moderate (a real walk-back "
        "on a supporting detail), strong (contradicts something central to "
        "their own stated defense), decisive (guts their entire account of "
        "what happened). Pick the lowest level the tension actually supports "
        "— do not round up because it feels notable."
    )


class SuspectNarrative(BaseModel):
    """The suspect's account held as ONE evolving story, updated each turn.
    This is deliberately separate from the atomic claims list: the claims
    list exists to check individual propositions against facts/policy, and
    decomposing the answer into atoms for that purpose throws away the
    connective tissue between turns. This model is where that connective
    tissue lives instead."""

    summary: str = Field(
        description="3-5 sentences: the suspect's account of what happened, "
        "as they have told it SO FAR across the whole interview, in their "
        "own logic — not whether the investigator believes it."
    )
    self_contradictions: list[SelfContradiction] = Field(
        default_factory=list,
        description="Only genuinely new tensions surfaced by THIS turn's "
        "answer against something said earlier. Do not re-list a tension "
        "already identified in a previous turn.",
    )
    stale_thread: Optional[str] = Field(
        default=None,
        description="Set only if the investigator's last 3+ questions have "
        "circled the same underlying point without the suspect adding "
        "anything genuinely new, even if worded differently each time. "
        "Name the specific topic in a few words (e.g. 'whether the suspect "
        "saw the transaction alert'). Leave null otherwise — this is not "
        "for a thread that is still producing movement."
    )


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
    future_verification_value: FutureVerificationValue = FutureVerificationValue.NONE
    notes: Optional[str] = None


class CaseLog(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)
    leads: list[InvestigationLead] = Field(default_factory=list)

    contradictions: list[str] = Field(default_factory=list)
    admissions: list[str] = Field(default_factory=list)
    active_defenses: list[str] = Field(default_factory=list)
    credibility_flags: list[str] = Field(default_factory=list)
    evasions: list[str] = Field(default_factory=list)

    unresolved_claims: list[str] = Field(default_factory=list)
    parked_threads: list[str] = Field(default_factory=list)
    exhausted_targets: list[str] = Field(default_factory=list)

    def investigator_view(self) -> dict:
        """Safe shared context for War Room/Speaker; hidden-truth findings are removed.

        `relation` is deliberately excluded from each claim's dump here: it's kept
        on ClaimRecord for later audit/debugging, but it says almost the same thing
        as `status` in different words, so showing both to War Room/Speaker would
        just be redundant tokens with no extra decision-making value for them.
        """
        return {
            "claims": [
                c.model_dump(exclude={"relation"})
                for c in self.claims if c.investigator_visible
            ],
            "leads": [lead.model_dump() for lead in self.leads],
            "contradictions": list(self.contradictions),
            "admissions": list(self.admissions),
            "active_defenses": list(self.active_defenses),
            "credibility_flags": list(self.credibility_flags),
            "evasions": list(self.evasions),
            "unresolved_claims": list(self.unresolved_claims),
            "parked_threads": list(self.parked_threads),
            "exhausted_targets": list(self.exhausted_targets),
        }


class WorldAction(BaseModel):
    action: str
    action_type: str = Field(description="physical_object, person_presence, completed_check, environment_change, or other")
    allowed: bool
    reason: str


class RealityGateResult(BaseModel):
    analysis_text: str = Field(
        description="Verbal/factual content that remains after blocked world-changing actions are removed"
    )
    actions: list[WorldAction] = Field(default_factory=list)
    blocked_message: Optional[str] = None

    @property
    def has_blocked_action(self) -> bool:
        return any(not action.allowed for action in self.actions)


class StrategistMove(BaseModel):
    # case_review is declared FIRST on purpose: structured output is generated
    # field by field in schema order, so this forces the model to actually
    # reason in prose about the whole case BEFORE it commits to target/tactic,
    # instead of picking a target first and writing a justification for it
    # afterward. See agents.strategist_prompt for what this must cover.
    case_review: str = Field(
        description="Think through the case so far in your own words before "
        "deciding anything: what is the suspect's actual central claim or "
        "defense, which currently-unresolved points already carry high "
        "strategic_value in the case log regardless of how long ago they "
        "were raised, and which of those is still genuinely untested. This "
        "is not a summary for its own sake — your target below must follow "
        "from what you conclude here. Every factual assertion you make here "
        "must trace back to something actually present in known_facts or the "
        "case log below — never state a document, record, signature, or "
        "outside check as existing or confirmed unless it is literally there. "
        "If something is unconfirmed, say it is unconfirmed."
    )
    target: str = Field(description="Free-form investigation thread to pursue next")
    tactic: str = Field(description="Allowed interview tactic")
    rationale: str = Field(
        description="Internal strategy rationale. Must name which War Room "
        "voice's read (Skeptic's push_now vs. Alternative's caution_move) "
        "most shaped this move, and why the other one was outweighed this "
        "turn — not a restatement of both, an actual adjudication."
    )
    expected_value: str = Field(default="medium", description="low, medium, or high")


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
    evidentiary_impact: EvidentiaryImpact = EvidentiaryImpact.NONE


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
    verification_score_delta: int = 0
    final_score: int = 0


class GameState(BaseModel):
    case: CaseFile
    transcript: list[dict] = Field(default_factory=list)
    case_log: CaseLog = Field(default_factory=CaseLog)

    score: int = 0
    tier: int = 1
    question_count: int = 0
    max_questions: int = 7

    scored_findings: list[str] = Field(default_factory=list)
    last_turn_delta: int = 0
    last_turn_usefulness: str = "unknown"
    last_world_notice: Optional[str] = None
    last_move: Optional[str] = None
    move_history: list[str] = Field(default_factory=list)

    # One SuspectDemeanor value per turn, read fresh from that turn's answer
    # only (see ExtractedClaims.demeanor). current_demeanor and the trend are
    # both derived from this list in code — never stored separately, so
    # there is nothing here for two code paths to disagree about.
    demeanor_history: list[str] = Field(default_factory=list)
    last_demeanor_cue: Optional[str] = None

    # Consecutive turns with no evidentiary gain and no substantive content
    # (a pure refusal/non-answer, or an answer flagged as evasion with
    # nothing to check). See game_logic.process_turn_scoring: this is what
    # lets sustained non-cooperation carry a real cost under the "Duty to
    # Cooperate" policy instead of being scoring-neutral, which previously
    # made silence the dominant strategy over even a caught liar.
    consecutive_stonewall: int = 0

    # The suspect's account held as one evolving story (see SuspectNarrative).
    # Updated every turn from the full transcript, not the atomic claims
    # list — this is what lets the investigator notice its own story-vs-story
    # contradictions and its own topic tunnel-vision, neither of which the
    # per-claim Checker pipeline can see by construction.
    narrative: Optional[SuspectNarrative] = None

    @property
    def current_demeanor(self) -> str:
        return self.demeanor_history[-1] if self.demeanor_history else "composed"


_ESCALATED_DEMEANORS = {"aggressive", "panicked"}
_CALM_DEMEANORS = {"composed", "cooperative"}


def demeanor_trend(history: list[str]) -> str:
    """Pure code derivation from stored per-turn demeanor reads — the LLM is
    never asked to judge the trend itself, only each turn's raw demeanor."""
    if len(history) < 2:
        return "opening move, no trend yet"
    current, previous = history[-1], history[-2]
    if current == previous:
        return f"holding steady at {current}"
    if current in _ESCALATED_DEMEANORS and previous not in _ESCALATED_DEMEANORS:
        return f"escalating toward {current}"
    if current in _CALM_DEMEANORS and previous not in _CALM_DEMEANORS:
        return f"cooling from {previous} to {current}"
    return f"shifting from {previous} to {current}"