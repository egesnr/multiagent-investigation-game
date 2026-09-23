"""
Core schemas for the open-world multi-agent investigation game.

The authored CaseFile is fixed world truth.
GameState is the shared interview context.
Hidden truth may be used by the Checker/Resolution agent, but it must not leak
into the War Room or Speaker unless the investigator has actually established it.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator



# Longest a model-authored free-text field may be before it is treated as
# broken output rather than an answer. Generous: the longest legitimate
# case_review seen in testing was around 1,200 characters.
MAX_FIELD_CHARS = 4000


def sane_text(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    """Truncate a runaway model field, and say so in the value itself.

    Kept visible rather than silent: a truncated field in a log is a signal
    that the model degenerated on that call, which is worth noticing.
    """
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return (
        value[:limit].rstrip()
        + f" …[truncated: model emitted {len(value)} characters]"
    )


def _truncating_validator(*fields: str):
    """Attach sane_text to the named fields of a model."""
    return field_validator(*fields, mode="after")(
        classmethod(lambda cls, v: sane_text(v) if isinstance(v, str) else v)
    )


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
    could_both_hold: str = Field(
        description="Before choosing a relation: describe any situation in "
        "which BOTH propositions are true at once, or write 'none' if there "
        "genuinely is not one. CONTRADICTS is available only when the honest "
        "answer is 'none'. Different numbers about different things are the "
        "trap this field exists to catch — what something cost and what was "
        "charged for it can both be true and differ, which is what an error "
        "IS. Measured without this field: a claim that a meal cost one amount "
        "was marked as contradicted by the settled card amount, and scored as "
        "a proven lie, when nothing available established the meal's price "
        "either way."
    )
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
    suggested_thread: Optional[str] = Field(
        default=None,
        description="ONE line of inquiry this claim opens, in a single short "
        "sentence — the one thing most worth pursuing, not a list of "
        "everything that could be checked. Null when the claim opens nothing "
        "new. Left undescribed, this field produced 393,000 characters of "
        "enumerated follow-ups in one call, which was 63% of that turn's "
        "running time.",
    )


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

    # Carried through from the Extractor (not re-decided by the Checker) so the
    # scoring key can group restatements of one topic together.
    subject: str = ""

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

    _keep_text_sane = _truncating_validator("quoted_evidence", "rationale", "subject", "suggested_thread")


class ClaimRecord(BaseModel):
    subject: str = ""
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

    _keep_text_sane = _truncating_validator("text", "rationale", "subject", "suggested_thread")


class ClaimNoveltyStatus(str, Enum):
    NEW = "new"
    REITERATED = "reiterated"
    UPDATED = "updated"


class ExtractedClaimItem(BaseModel):
    # Declared before `text` on purpose: naming what the claim is ABOUT before
    # writing the claim forces the model to locate it on the existing record
    # rather than compose a fresh sentence and then try to remember whether
    # something like it was already said. This is also what makes scoring
    # dedup possible at all — see game_logic._claim_key.
    subject: str = Field(
        description="What this claim is about, as a short noun phrase: the "
        "topic slot it occupies on the record. Not the assertion itself — "
        "two claims that disagree share one subject. If a subject already on "
        "record covers this claim, reuse that wording EXACTLY. Invent a new "
        "subject only when nothing on record covers it."
    )
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
    # The dodge verdict is declared BEFORE the claims for the usual reason:
    # deciding whether the question was answered, while the question is still
    # the thing in mind, is more reliable than extracting content first and
    # then reverse-engineering whether it was responsive.
    propositions: list[str] = Field(
        default_factory=list,
        description="Every distinct thing this answer asserts, one per entry, "
        "before you build any claims. A run-on answer often carries several, "
        "and they can sit badly against each other — list them all anyway, "
        "including the ones that damage the suspect and the ones that help "
        "them. Measured without this field: an answer containing an excuse AND "
        "an admission of awareness yielded only the excuse, and the admission "
        "never reached the record at all. Build the claims below from this "
        "list, not from the sentence.",
    )
    claims: list[ExtractedClaimItem] = Field(default_factory=list)
    new_defenses: list[str] = Field(default_factory=list)


class SelfContradiction(BaseModel):
    """A tension between two of the SUSPECT'S OWN statements, found by reading
    the whole account as one story — not a claim checked against an authored
    fact. This is the thing atomic claim-by-claim checking structurally
    cannot find on its own: it requires holding two things said in different
    turns in mind at once and noticing they don't sit together."""

    subject: str = Field(
        description="What the tension is ABOUT, as a short noun phrase — the "
        "topic slot it occupies on the record, the same way an extracted "
        "claim carries one. Declared before claim_text on purpose: name the "
        "topic before writing the sentence, so a tension already on record "
        "gets recognised instead of re-described in fresh words. If a "
        "contradiction about this subject has already been flagged, you are "
        "restating it, not finding a new one: every rewording of one tension "
        "belongs to the subject it is about."
    )
    claim_text: str = Field(
        description="A single, self-contained statement of the tension for "
        "the case log and the investigator, e.g. 'The suspect's account of "
        "noticing the alert is internally inconsistent: they first denied "
        "seeing it, then said they saw it but dismissed it.'"
    )
    earlier_statement: str = Field(description="The earlier statement, quoted or closely paraphrased.")
    later_statement: str = Field(description="The later statement that sits badly against it.")
    why_incompatible: str = Field(
        description="State what must be true for the earlier statement to hold, "
        "what must be true for the later one, and why both cannot hold at "
        "once. Declared before the impact rating on purpose: write this "
        "sentence first, and if you cannot write it, there is no "
        "contradiction — leave the item out entirely rather than rating it. "
        "Two statements that restate the same thing, or that both follow from "
        "the same premise, are not in tension however differently they are "
        "worded."
    )
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
    unfalsifiable_account: Optional[SelfContradiction] = Field(
        default=None,
        description="Set ONCE, and only once, when the suspect's account has "
        "become one that nothing in it can be checked by anyone — every "
        "element conveniently beyond verification: cannot recall, cannot "
        "name, no documentation, the only witness unavailable, the only "
        "record the one thing they say proves them right. This is not the "
        "same as a claim merely being unverified; it is the shape of the "
        "WHOLE account, and it is a recognised signature of fabrication. "
        "The test is whether a person telling the truth about these events "
        "would really be unable to offer a single checkable detail. Use the "
        "same fields as a self-contradiction: claim_text states the pattern "
        "plainly, earlier_statement and later_statement quote two of the "
        "unverifiable elements, and evidentiary_impact rates how damning the "
        "pattern is. Leave null while the suspect is still offering things "
        "that could be checked, and leave null on every turn after you have "
        "already flagged it once.",
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

    _keep_text_sane = _truncating_validator("topic", "source_claim", "notes")


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


class SpeakerLine(BaseModel):
    """What the investigator actually says, plus the sources behind it.

    `grounding` is declared FIRST on purpose, same trick as case_review on
    InvestigatorMind: structured output is generated in schema order, so the
    model must name a source for each specific claim BEFORE it writes the
    sentence containing it. Enumerating forbidden categories in the prompt
    kept failing — every time one category was closed (inventing documents,
    then inventing what the suspect saw, then inventing how card systems
    work) the next invention simply appeared in a category nobody had listed
    yet. Requiring a source for every specific claim closes all of them at
    once, including the ones nobody has thought of.
    """

    grounding: list[str] = Field(
        default_factory=list,
        description="Every specific factual claim about THIS case that your "
        "line will state or imply, each paired with where it came from, "
        "written as 'claim — source'. A source is one of: a known fact you "
        "were given, an entry in the visible case log, or the suspect's own "
        "words in the transcript (quote the phrase). If you cannot name a "
        "real source for something, you may not say it — either drop it, or "
        "rewrite it as a general statement about how things usually work "
        "('receipts are normally handed over' rather than 'you were handed a "
        "receipt'). An empty list is correct for a line that asserts nothing "
        "specific, such as a pure question, demand, or challenge.",
    )
    line: str = Field(
        description="The words you actually say to the suspect. Every "
        "specific claim it makes must appear in grounding above."
    )


class InvestigatorMind(BaseModel):
    """One read of the whole case, replacing four separate calls (narrative
    synthesis, skeptic, alternative hypothesis, strategist).

    Those four were split by INFORMATION as well as by task, and each saw a
    different slice of the world: the narrative agent read the entire
    transcript but was never shown a single fact or policy rule, the skeptic
    had facts but no policy, and the strategist — the one actually choosing
    the next question — had neither. So no agent in the room could notice
    a conclusion that needed an authored fact, a policy rule and the suspect's
    answer at the same time, because no agent held all three.

    Field order is the reasoning order. unused_facts and inferences come
    before every judgment for the same reason case_review precedes target:
    structured output is generated in schema order, so the model must walk
    the evidence and draw conclusions from it BEFORE it is allowed to pick
    what to ask.
    """

    # --- 1. did they answer the question? ---
    # First, because everything after it depends on whether this turn produced
    # an account or an evasion, and because judging it needs the whole
    # interview rather than one answer in isolation.
    what_was_asked: str = Field(
        default="",
        description="In one line, what your last question actually demanded — "
        "what would have to be in a reply for it to count as answered. Not a "
        "restatement of the question's wording.",
    )
    what_they_offered: str = Field(
        default="",
        description="State what the answer actually puts forward, in its own "
        "terms, before you judge it. Write it as they meant it, not as you "
        "would rebut it. Declared before the verdict because the two are easy "
        "to collapse: an account you find false is still an account.",
    )
    was_it_given: str = Field(
        default="",
        description="Does what they offered address what you asked? Yes or no, "
        "and why. This is a question about RESPONSIVENESS ONLY. A lie is an "
        "answer. A hostile, vague or partial reply that engages with the "
        "question is an answer. Disbelieving it, or being able to disprove it, "
        "does not make it a non-answer — that is what the rest of the pipeline "
        "is for. Only a reply that changes the subject, complains about the "
        "question, or tells you to look it up yourself is a non-answer.",
    )
    subject_dodged: Optional[str] = Field(
        default=None,
        description="When it was not given: the subject of what you asked, as "
        "a short noun phrase, at the granularity of the question — not the "
        "case's overall topic. A question about a receipt and a question about "
        "a calendar are two subjects even when both concern one transaction. "
        "Reuse a subject already on record only when it is the same question "
        "being put again. Null when the question was answered.",
    )
    dodge_settles_fact: bool = Field(
        default=False,
        description="When it was not given: true only if you asked them to "
        "PRODUCE or ACCOUNT FOR something that either exists or does not — a "
        "receipt, a name, a date, a document, a witness. Refusing to produce "
        "it is evidence it does not exist. False when you asked them to admit "
        "wrongdoing or say what they knew or intended: silence can never "
        "establish a state of mind, and a refusal to confess is never a "
        "confession. If unsure, false.",
    )

    # --- 2. read the whole picture ---
    account_summary: str = Field(
        description="3-5 sentences: the suspect's account of what happened, as "
        "they have told it so far across the whole interview, in their own "
        "logic — not whether you believe it."
    )
    unused_facts: list[str] = Field(
        default_factory=list,
        description="Go through the known facts one by one and list every fact "
        "that has NOT yet been put to the suspect in any question. Just the "
        "fact ids or short labels. This is a checklist, not a judgment: if a "
        "fact is sitting there unused, it belongs in this list even if you do "
        "not intend to use it this turn. Empty only when genuinely all of them "
        "have been raised."
    )
    inferences: list[str] = Field(
        default_factory=list,
        description="What follows from combining the facts, the policy rules "
        "and what the suspect has actually said — conclusions nobody has "
        "stated yet. Cover both halves: what follows about the SUSPECT, and "
        "what follows about ANYONE OR ANYTHING ELSE involved — what another "
        "person, business or system would have done or noticed if their "
        "account were true, and whether the suspect behaved like someone for "
        "whom it was true. The second half is the one that gets forgotten, and "
        "it is usually where an account comes apart. Do the arithmetic where "
        "it bites. Each entry must be derivable from known_facts, the policy, "
        "or the suspect's own words — never from something you assume or would "
        "like to be true. Empty if nothing genuinely follows."
    )

    # --- 3. findings that score ---
    self_contradictions: list[SelfContradiction] = Field(
        default_factory=list,
        description="Only genuinely new tensions surfaced by THIS turn's answer "
        "against something the suspect said earlier. Two of THEIR statements, "
        "never their statement against a record — the Checker owns that "
        "comparison and has already made it. Do not re-list a tension "
        "identified in a previous turn.",
    )
    unfalsifiable_account: Optional[SelfContradiction] = Field(
        default=None,
        description="Set ONCE, ever, when the account has become one where "
        "nothing in it can be checked by anyone: cannot recall, cannot name, no "
        "documentation, the only witness unavailable. Not the same as a claim "
        "merely being unverified — it is the shape of the WHOLE account. Null "
        "while they are still offering checkable detail, and null on every turn "
        "after it has been flagged once.",
    )
    stale_thread: Optional[str] = Field(
        default=None,
        description="Set only if the last 3+ questions circled the same "
        "underlying point without the suspect adding anything new, however "
        "differently worded. Name the topic in a few words. Null if the thread "
        "is still producing movement.",
    )

    # --- 4. the decision ---
    # The two war-room reads are NOT fields here. They arrive as input, from
    # two calls that could not see each other's answer. One model writing both
    # sides in sequence is not a debate — it already knows which side it wants,
    # and can write a weak innocent reading to justify it.
    case_review: str = Field(
        description="Having weighed both readings above, think through the case "
        "before deciding anything: the suspect's actual central claim, which "
        "unresolved points carry high strategic value regardless of how long "
        "ago they were raised, and which of those is still genuinely untested. "
        "Your target must follow from this. Every factual assertion here must "
        "trace to known_facts, the policy, or the case log — if something is "
        "unconfirmed, say so."
    )
    repetition_check: str = Field(
        description="Restate what EVERY move in move_history was actually "
        "asking — the underlying question, not the wording — then check "
        "whether your planned target asks any one of them again in different "
        "clothes. Judge by meaning: a question from turn 1 is exactly as "
        "repeated as one from last turn. If it repeats, pick a different "
        "thread and say so here."
    )
    target: str = Field(
        description="Free-form thread to pursue next, carrying both what you "
        "are going after and how you intend to come at it, as you would tell a "
        "partner ('corner him on the shifting story about the notifications'). "
        "Any specific you reference — a number, a document, something the "
        "suspect saw — must be something they actually said or a fact from "
        "known_facts. Never invent a specific because it seems a likely "
        "inference."
    )
    innocent_reading_won: bool = Field(
        description="True if the Alternative's read actually changed what you "
        "are about to do — softened the target, redirected it, or stopped you "
        "pressing something. False if you went with the Skeptic. Answer for "
        "what you actually did, not for what sounds balanced."
    )
    rationale: str = Field(
        description="Why this move over the other side's. Name which read won "
        "and why the other was outweighed — an adjudication, not a restatement "
        "of both."
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
    max_questions: int = 8

    scored_findings: list[str] = Field(default_factory=list)

    # Distinct subjects the suspect has refused to address, in the order they
    # were first dodged. The stall charge escalates down this list, so refusing
    # broadly costs far more than refusing the same thing repeatedly — which is
    # also the only version an investigator would actually care about.
    dodged_subjects: list[str] = Field(default_factory=list)
    # How many times each has been dodged. At two, the investigator states what
    # the silence will be recorded as and the subject closes to further asking.
    dodge_counts: dict[str, int] = Field(default_factory=dict)
    # Reversible half: the fact taken as established because the suspect would
    # not address it. Released if they later answer. The stall charge is not.
    dodge_fact_points: dict[str, int] = Field(default_factory=dict)
    # Human-readable twin of scored_findings, for the investigator's own
    # context. The keys above are internal and unreadable; these are the
    # subjects themselves, so the room can be told plainly which ground is
    # already banked and will not pay again.
    banked_subjects: list[str] = Field(default_factory=list)

    # Claim text -> provisional suspicion points awarded in the room for an
    # unverified defense. Resolution refunds these if verification confirms
    # the claim was true, so a suspect is never permanently punished for an
    # excuse that turns out to be honest.
    provisional_findings: dict[str, int] = Field(default_factory=dict)
    last_turn_delta: int = 0
    last_turn_usefulness: str = "unknown"
    last_world_notice: Optional[str] = None
    last_move: Optional[str] = None
    move_history: list[str] = Field(default_factory=list)

    # Set once the narrative agent has flagged the unfalsifiable-account
    # pattern. Its memory is rebuilt every turn and the only history it
    # receives is the contradictions list, which this pattern deliberately is
    # not part of — so without this it re-flagged the same pattern on a later
    # turn with an extra clause appended, and scored it twice.
    unfalsifiable_flagged: bool = False

    # One entry per turn: the sources the Speaker named for that turn's line
    # (see SpeakerLine.grounding). Kept on the state, not just the debug log,
    # so a saved session can be audited afterwards for claims that were
    # asserted without a real source — including citations the model invented.
    grounding_history: list[list[str]] = Field(default_factory=list)

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