"""
Interview agents.

Flow:
Extractor -> Checker -> Shared Context -> War Room -> Lead Strategist -> Speaker

Knowledge boundary:
- Checker, War Room, and Speaker all see only investigator-known facts.
- Hidden authored truth is not passed to any interview-time agent. It exists in
  the case file for the Resolution Agent (post-interview) to reason from
  directly, not for the interview pipeline to check claims against.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models import (
    CaseFile,
    CheckResult,
    EvidenceAssessment,
    ExtractedClaimItem,
    ExtractedClaims,
    InvestigativeSignificance,
    ClaimStatus,
    ClaimType,
    EvidentiaryImpact,
    FindingBasis,
    FutureVerificationValue,
    GameState,
    RealityGateResult,
    StrategistMove,
    EvidenceRelation,
    demeanor_trend,
    format_facts,
)

load_dotenv()

MODEL_NAME = os.environ.get("GAME_MODEL", "gemini-3.1-flash-lite")


def get_llm(temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
    )




reality_gate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Reality/Action Gate for an open-world interview game.

The suspect may SAY anything, including lies, strange explanations, claims about
people elsewhere, or claims that an object exists outside the room. Do not judge
those statements as true or false here.

But text cannot magically change the physical world. Detect attempted immediate
world-changing actions such as:
- producing/handing over a physical object,
- making a new person physically present,
- declaring that investigators already completed a check,
- changing the room/environment or an already established external result.

An action is allowed only if the required object/person/result is already present
in WORLD STATE below. A verbal claim like "I have a receipt at home" is speech,
not a blocked action. A mixed answer like "the restaurant gave me a receipt; here
it is" should keep the verbal claim in analysis_text while blocking only the
physical handover if the receipt is not available.

Return analysis_text containing all usable verbal/factual content with blocked
action language removed. If the answer contains only a blocked action, return an
empty analysis_text and a short blocked_message explaining the world limitation.
Never add facts that the player did not say.

WORLD STATE:
Present people: {present_people}
Objects physically available to suspect in room: {room_objects}
Already established investigator-visible facts: {known_facts}
Recent transcript: {transcript}"""
    ),
    ("human", "Suspect input:\n{answer}"),
])


def run_reality_gate(state: GameState, answer: str) -> RealityGateResult:
    chain = reality_gate_prompt | get_llm(0).with_structured_output(RealityGateResult)
    return chain.invoke({
        "present_people": state.case.present_people,
        "room_objects": state.case.room_objects or ["none authored"],
        "known_facts": _known_facts(state),
        "transcript": state.transcript[-8:],
        "answer": answer,
    })


extractor_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract meaningful factual claims from the suspect's latest answer.

Do not invent details.

ATOMIC CLAIM RULE:
- Split compound answers into separate factual claims.
- One claim should contain one proposition only.
- Do not bundle an explanation with the underlying fact.
- Keep each claim short, specific, and self-contained.

RULE 1 — CONTEXT SYNTHESIS:
Short or partial answers (e.g. "yes", "no", "I did") are meaningless on their own.
Combine them with the previous_question to produce one complete, standalone claim
that states exactly what was confirmed or denied. Never extract a bare "yes"/"no"
as the claim text itself.

RULE 2 — STRICT MODALITY:
Extract exact meanings only.
- Do not upgrade a general habit, possibility, or policy statement into a specific
  admission about this suspect.
- Do not take rhetorical, boastful, or status-flexing statements (self-praise,
  claims of importance, exaggerated value statements) and convert them into a
  literal checkable fact. If a statement is rhetorical/emotional posturing rather
  than a factual assertion, set checkable=false or omit it as a claim entirely.

RULE 3 — STATE DIFFING:
You will be given known_claims already on record. For every extracted claim,
compare it against known_claims and set status:
- NEW: not previously stated, in substance.
- REITERATED: restates a known_claim with the same meaning, even if reworded.
- UPDATED: revises or replaces a known_claim with a materially different version
  (e.g. suspect changes their account of what happened).
Judge by meaning, not exact wording.

RULE 4 — ADMISSION VS. DEFENSE SEPARATION:
For every claim, set is_defense:
- is_defense=false: a plain, undisputed factual admission (what happened).
- is_defense=true: an excuse, alternative explanation, unverified alibi, or any
  claim offered to justify/explain away suspicion — even if it is concrete and
  checkable (e.g. "the charge was a billing error" is a checkable claim AND a
  defense; mark both is_defense=true and checkable=true).
This is a checkable-vs-not distinction, not a claims-vs-defenses bucket: checkable
defenses still belong in claims so they can be verified.

Statements with no factual content at all (pure opinion, denial of intent framed
as character, emotional appeals) go in new_defenses instead of claims.

RULE 5 — LITERAL ATTRIBUTION UNDER AMBIGUITY:
If a claim's agent, referent, or meaning could reasonably be read more than one
way from the suspect's actual words, do not resolve it by choosing the more
specific, more active, or more damaging interpretation. Instead:
- Write `text` as the weakest, most literal reading that both interpretations
  would still support (e.g. "the suspect became aware of the discrepancy in
  connection with an investigation" rather than asserting who conducted it).
- Set ambiguous=true.
Only set ambiguous=false when the suspect's words support exactly one reading.

RULE 6 — SUSPECT-ONLY GROUNDING:
- Every claim must be grounded only in the suspect's own words in the CURRENT
  answer. Never construct a claim from the investigator's question phrasing,
  even if the suspect's answer is responding to it. If the investigator's
  question contains a phrase (e.g. "business meeting all day") and the suspect
  does not repeat or affirm it, that phrase must not appear in any extracted
  claim this turn.
- Do not manufacture a claim asserting the negation of something the suspect
  did not explicitly deny. Describing new activity ("I slept, then worked")
  is not the same as an explicit denial ("I was not in a meeting"). Only
  extract a denial claim if the suspect's words actually deny it.
- Do not re-emit a claim from a previous turn as if newly said this turn
  unless the suspect's current answer actually restates it in some form.
  known_claims is for comparison/deduplication only — it is never a source of
  new claims for the current turn.

RULE 7 — DEMEANOR READING:
Separately from claim content, read how the suspect is coming across in THIS
answer only (ignore earlier turns entirely — that history is tracked
elsewhere). Choose exactly one:
- composed: calm, in control, answering directly.
- cooperative: forthcoming, volunteering detail, engaged with the process.
- defensive: justifying, minimizing, rushing to explain rather than answer.
- evasive: dodging, non-answers, changing the subject, answering a different
  question than the one asked.
- aggressive: hostile, challenging the investigator's right to ask, raised
  tone, attacking rather than answering.
- panicked: flustered, self-contradicting within the same answer,
  over-explaining, visible distress.
Base this purely on tone/behavior in the wording actually used, not on
whether the content sounds truthful. A calm liar is still composed; an
innocent person who is furious at being accused is still aggressive.
Give a one-phrase demeanor_cue naming the specific tell, or leave it empty
only if the answer is too short/flat to show one.
"""
    ),
    (
        "human",
        "Recent interview context:\n{transcript}\n\n"
        "Previous question:\n{question}\n\n"
        "Suspect answer:\n{answer}\n\n"
        "Known claims already on record:\n{known_claims}"
    ),
])


def run_extractor(
    question: str,
    answer: str,
    known_claims: list[str] | None = None,
    transcript: list[dict] | None = None,
) -> ExtractedClaims:
    chain = extractor_prompt | get_llm(0).with_structured_output(ExtractedClaims)
    return chain.invoke({
        "question": question,
        "answer": answer,
        "known_claims": known_claims or ["- None yet"],
        "transcript": (transcript or [])[-8:],
    })


class EvidenceAssessmentList(BaseModel):
    results: list[EvidenceAssessment] = Field(default_factory=list)


# The LLM only ever decides `relation` (real reasoning) and `is_admission`
# (a separate yes/no on direct confession). verification_status is a pure
# derived label computed here in code so it can never drift out of sync with
# relation, and so we stop paying LLM output tokens to re-derive the same
# judgment in different words.
RELATION_TO_STATUS = {
    EvidenceRelation.ENTAILS: ClaimStatus.SUPPORTED,
    EvidenceRelation.CONTRADICTS: ClaimStatus.CONTRADICTED,
    EvidenceRelation.NEUTRAL: ClaimStatus.UNVERIFIED,
}


class InvestigativeSignificanceList(BaseModel):
    results: list[InvestigativeSignificance] = Field(default_factory=list)


evidence_checker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Evidence Checker.

Your only job is to determine the evidentiary status of each extracted suspect claim.

Do not assess:
- strategic value
- future verification value
- evidentiary impact
- interrogation strategy
- investigative importance

For every claim, reason in this exact order.

1. CLAIM

Preserve the exact suspect claim being evaluated.

2. CLAIM PROPOSITION

State precisely what the claim asserts.

Do not broaden, reinterpret, or strengthen the claim.

3. EVIDENCE PROPOSITION

Identify the most relevant available evidence and state precisely what that evidence establishes.

Do not broaden what the evidence proves.

If no available evidence directly bears on the claim, use:

"none"

4. EVIDENCE RELATION

Compare the evidence proposition against the claim proposition.

Choose exactly one relation:

ENTAILS
The evidence logically establishes that the claim is true.

CONTRADICTS
The evidence logically establishes that the claim is false.

NEUTRAL
The evidence establishes neither the claim nor its negation.

Use NEUTRAL when:
- the evidence is merely related to the claim,
- the evidence creates suspicion but does not prove the claim false,
- there is a discrepancy that does not logically resolve the claim,
- information is missing,
- further verification is required,
- the evidence concerns a different property of the same event.

A contradiction requires evidence that logically establishes the negation of the exact claim proposition.

Do not treat:
- different numbers,
- different records,
- different stages of an event,
- suspicious circumstances,
- missing documentation,
- lack of corroboration,
- or evidence about a related issue

as contradiction unless that evidence logically proves the exact claim false.

5. ADMISSION CHECK

Set is_admission=true only if the suspect explicitly and directly confesses to a
materially damaging fact in their own words.

Do not set is_admission=true merely because the suspect states an ordinary
background fact, or because the evidence relation happens to be CONTRADICTS or
ENTAILS. This is judged independently of the evidence relation — it is only
true for a direct, voluntary confession.

6. BASIS

Choose the source that actually resolves or bears on the claim:

- investigator_evidence:
  investigator-visible evidence directly bears on the claim.

- story_history:
  prior suspect statements directly support or contradict the claim.

- policy:
  investigator-visible facts directly establish a policy conclusion.

- unresolved:
  no evidentiary source currently bears on the claim, or it requires later
  verification.

GENERAL RULES

- Unknown is not false.
- Absence of evidence is not contradiction.
- Suspicion is not contradiction.
- A discrepancy is not automatically contradiction.
- Related evidence is not automatically contradictory evidence.
- Do not invent evidence.
- Do not infer facts that are not established.
- Do not use strategic usefulness to influence evidentiary status.
- Do not use future verifiability to influence evidentiary status.
- Do not use how suspicious a claim sounds to influence evidentiary status.

INVESTIGATOR-KNOWN CASE FACTS:
{known_facts}

POLICY:
{policy}

INVESTIGATOR-VISIBLE CASE LOG:
{case_log}

RECENT TRANSCRIPT:
{transcript}
"""
    ),
    (
        "human",
        """Evaluate these latest claims:

{claims}
"""
    ),
])

significance_checker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Investigative Significance Checker.

The Evidence Checker has already determined each claim's evidentiary status.
Do not override or reinterpret that status, basis, visibility, proposition analysis,
or compatibility judgment.

Your only job is to assess investigative significance.

For every supplied assessment determine:
- claim_type: choose only BACKGROUND (plain neutral fact, no persuasive/defensive
  function) or OTHER (checkable claim that is neither neutral background nor
  self-serving, e.g. it implicates a third party). Do not choose ADMISSION,
  DEFENSE, or CONTRADICTION — those are determined automatically from upstream
  fields and any value you set for them will be overwritten.
- risk_profile
- strategic_value
- future_verification_value
- evidentiary_impact
- suggested_thread

STRICT RULES:
- Preserve the Evidence Checker's verification_status exactly as given.
- Preserve basis exactly as given.
- Do not turn an unresolved claim into a contradiction.
- fact_contradiction, story_contradiction, and policy_breach will all be
  overwritten downstream to false whenever verification_status is unverified
  — set them to your best guess, but do not spend extra effort second-guessing
  basis to influence them.
- If verification_status is unverified:
  - fact_contradiction=false
  - story_contradiction=false
  - policy_breach=false
  - evidentiary_impact=none
- Suspiciousness, implausibility, or future checkability are not current evidence.
- A claim may have high future_verification_value while evidentiary_impact=none.
- Only established contradictions, damaging admissions, or established damaging policy facts may carry current incriminating evidentiary impact.

EVIDENTIARY IMPACT RUBRIC (apply exactly one, do not blend):
- weak: a minor inconsistency that does not touch the core allegation or the
  suspect's own stated defense (e.g. a small detail about timing or location
  that isn't central to guilt).
- moderate: contradicts a supporting detail that strengthens suspicion but
  does not by itself undermine the suspect's central defense.
- strong: contradicts a claim that is central to the suspect's own stated
  defense or account (e.g. the suspect's own number, name, or explanation is
  directly disproven by known evidence).
- decisive: contradicts the core allegation itself, or disproves the
  suspect's entire defense, leaving no plausible innocent reading.
Pick the lowest level that the evidence actually supports. Do not round up
because a claim feels important; do not round down because a claim feels
uncomfortable to score highly.

CERTAINTY CEILING: each known fact is tagged [weight=..., certainty=...].
certainty=fast means a hard record (a settled transaction, a logged
reservation) capable of supporting any impact level up to decisive.
certainty=slow means the investigator's knowledge here is a review finding,
pattern, or inference rather than a primary record — evidence resting only on
a certainty=slow fact can never be scored above moderate, no matter how
central the claim looks, because the underlying knowledge isn't airtight
enough to be decisive on its own.
- Do not invent evidence.
- Do not invent policy conditions that are not established.
- suggested_thread should be a verification/investigation thread, not a scripted interview question.

INVESTIGATOR-KNOWN FACTS:
{known_facts}

POLICY:
{policy}

VISIBLE CASE LOG:
{case_log}

RECENT TRANSCRIPT:
{transcript}

EVIDENCE ASSESSMENTS:
{assessments}
"""
    ),
    ("human", "Assess the investigative significance of each evidence assessment above. Return one result per assessment in the same order."),
])


def _merge_checker_results(
    evidence_results: list[EvidenceAssessment],
    significance_results: list[InvestigativeSignificance],
    ambiguous_map: dict[str, bool] | None = None,
    defense_map: dict[str, bool] | None = None,
) -> list[CheckResult]:
    """Merge both checker stages while enforcing non-negotiable evidence invariants."""
    significance_by_claim = {r.claim.strip(): r for r in significance_results}
    merged: list[CheckResult] = []
    ambiguous_map = ambiguous_map or {}
    defense_map = defense_map or {}

    for index, evidence in enumerate(evidence_results):
        significance = (
            significance_results[index]
            if index < len(significance_results)
            else significance_by_claim.get(evidence.claim.strip())
        )
        if significance is None:
            significance = InvestigativeSignificance(claim=evidence.claim)

        impact = significance.evidentiary_impact
        if evidence.verification_status == ClaimStatus.UNVERIFIED:
            impact = EvidentiaryImpact.NONE

        risk_profile = significance.risk_profile.model_copy(deep=True)
        # fact_contradiction/story_contradiction are fully determined by
        # verification_status + basis, not a free LLM judgment: previously
        # only the false-positive direction was cleared (non-contradicted ->
        # both False), leaving the true-positive direction open to silent
        # drift (a CONTRADICTED claim could still show no contradiction flag
        # at all, or the wrong one of the two). Force both directions here,
        # same pattern as claim_type above.
        if evidence.verification_status == ClaimStatus.CONTRADICTED:
            risk_profile.fact_contradiction = evidence.basis in (
                FindingBasis.INVESTIGATOR_EVIDENCE, FindingBasis.POLICY
            )
            risk_profile.story_contradiction = evidence.basis == FindingBasis.STORY_HISTORY
        else:
            risk_profile.fact_contradiction = False
            risk_profile.story_contradiction = False

        # policy_breach means an established (non-unresolved) fact actually
        # violates a policy rule. Same reasoning as fact/story contradiction
        # above: an UNVERIFIED claim cannot itself establish a breach, so this
        # is forced false in code instead of trusting Stage 2 to remember the
        # rule on every call.
        if evidence.verification_status == ClaimStatus.UNVERIFIED:
            risk_profile.policy_breach = False

        # claim_type is forced from upstream fields wherever they already
        # settle the answer, so two independent LLM calls can never disagree
        # about the same underlying fact (same pattern as the
        # investigator_visible/hidden_truth fix above).
        if evidence.verification_status == ClaimStatus.ADMITTED:
            claim_type = ClaimType.ADMISSION
        elif evidence.verification_status == ClaimStatus.CONTRADICTED:
            claim_type = ClaimType.CONTRADICTION
        elif defense_map.get(evidence.claim.strip(), False):
            claim_type = ClaimType.DEFENSE
        else:
            claim_type = significance.claim_type

        merged.append(CheckResult(
            fact_id=evidence.fact_id,
            risk_profile=risk_profile,
            quoted_evidence=evidence.claim,
            rationale=evidence.rationale,
            basis=evidence.basis,
            # Always True: see the note on EvidenceAssessment in models.py.
            investigator_visible=True,
            relation=evidence.relation,
            claim_type=claim_type,
            verification_status=evidence.verification_status,
            strategic_value=significance.strategic_value,
            future_verification_value=significance.future_verification_value,
            evidentiary_impact=impact,
            suggested_thread=significance.suggested_thread,
            ambiguous=ambiguous_map.get(evidence.claim.strip(), False),
        ))

    return merged


def run_checker(
    case: CaseFile,
    claims: list[str],
    transcript: Optional[list[dict]] = None,
    case_log=None,
    ambiguous_map: dict[str, bool] | None = None,
    defense_map: dict[str, bool] | None = None,
) -> list[CheckResult]:
    if not claims:
        return []

    visible_facts = case.visible_facts("investigator_start")
    known_facts = format_facts(visible_facts)
    policy = "\n".join(f"- {p}" for p in case.policy_rules) or "- None"
    visible_case_log = case_log.investigator_view() if case_log else {}
    recent_transcript = (transcript or [])[-12:]

    evidence_chain = (
        evidence_checker_prompt
        | get_llm(0).with_structured_output(EvidenceAssessmentList)
    )
    evidence_output = evidence_chain.invoke({
        "known_facts": known_facts,
        "policy": policy,
        "case_log": visible_case_log,
        "transcript": recent_transcript,
        "claims": claims,
    })
    # Stage 1 is only ever shown investigator-visible facts now (hidden authored
    # truth is no longer passed in at all), so a valid fact_id can only be one
    # of these — anything else is a hallucination.
    valid_fact_ids = {f.id for f in visible_facts}
    for r in evidence_output.results:
        if r.fact_id and r.fact_id not in valid_fact_ids:
            print(
                f"[WARNING] fact_id '{r.fact_id}' on claim '{r.claim}' does not "
                "match any authored fact id in this case. Clearing it so the "
                "scoring dedup key falls back to text-based matching instead of "
                "trusting a possibly-hallucinated id."
            )
            r.fact_id = None
        r.verification_status = (
            ClaimStatus.ADMITTED if r.is_admission
            else RELATION_TO_STATUS[r.relation]
        )
        print(
            "[EVIDENCE RELATION]",
            r.claim,
            "| claim_prop =", r.claim_proposition,
            "| evidence_prop =", r.evidence_proposition,
            "| relation =", r.relation.value,
            "| is_admission =", r.is_admission,
            "| verification_status =", r.verification_status.value,
        )

    if not evidence_output.results:
        return []

    significance_chain = (
        significance_checker_prompt
        | get_llm(0).with_structured_output(InvestigativeSignificanceList)
    )
    significance_output = significance_chain.invoke({
        "known_facts": known_facts,
        "policy": policy,
        "case_log": visible_case_log,
        "transcript": recent_transcript,
        "assessments": [r.model_dump() for r in evidence_output.results],
    })

    return _merge_checker_results(
        evidence_output.results,
        significance_output.results,
        ambiguous_map=ambiguous_map,
        defense_map=defense_map,
    )


class EvidenceView(BaseModel):
    established: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    in_room_lever: str = Field(
        description="The single unresolved point that further QUESTIONING "
        "right now could still move. Never suggest fetching a document, "
        "calling a witness, or checking a record — that is what leads/"
        "parked_threads are for, not something this turn's tactic can do."
    )


class SkepticView(BaseModel):
    strongest_pressure_point: str
    behavioral_read: str = Field(
        description="What the suspect's demeanor and demeanor_cue this turn "
        "actually suggest — calculated evasion, rehearsed composure, genuine "
        "panic, etc. Ground this in the demeanor signal given, not a fresh "
        "guess from the transcript."
    )
    diversion_risk: str
    push_now: str = Field(
        description="The single most aggressive in-room move to make next "
        "turn — a question or confrontation, not an outside check."
    )


class AlternativeView(BaseModel):
    strongest_innocent_reading: str
    fairness_risk: str
    caution_move: str = Field(
        description="The single in-room move that fairly tests the innocent "
        "reading without assuming guilt — a question, not an outside check."
    )


class WarRoomBundle(BaseModel):
    evidence: EvidenceView
    skeptic: SkepticView
    alternative: AlternativeView


def _known_facts(state: GameState) -> str:
    return format_facts(state.case.visible_facts("investigator_start"))


evidence_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Evidence Analyst in an internal war room.
Separate established information from unresolved claims.
Do not infer hidden truth and do not invent evidence.
Your job is bookkeeping, not persuasion: give the cleanest possible map of
what stands and what doesn't, so the Skeptic and Alternative voices below
can argue over what to do about it. Do not recommend a next move yourself —
that is their job, not yours."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Suspect demeanor this turn: {demeanor} ({demeanor_cue})
Demeanor trend: {demeanor_trend}

Questions remaining: {remaining}
Current case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}
Recent strategy moves: {move_history}"""
    ),
])


skeptic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Skeptical Investigator in an internal war room — the
voice arguing to press harder, right now. Find the strongest legitimate
pressure point: established contradictions, policy admissions, material
unsupported explanations, evasion, or diversion. Read the suspect's demeanor
and demeanor_cue as a real signal, not decoration — a composed suspect
sitting on a strong contradiction is stalling; a suddenly panicked one just
got close to something.

Do not assume an unresolved claim is false.
Do not invent evidence.
Do not chase irrelevant side stories just because they are new.
Do not propose fetching a document, calling a witness, or checking a record —
propose what to ASK or how to CONFRONT, right now, in this room.
You are arguing a position, not filing a status report: be willing to
disagree with a more cautious read of the same facts."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Suspect demeanor this turn: {demeanor} ({demeanor_cue})
Demeanor trend: {demeanor_trend}

Questions remaining: {remaining}
Current case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}
Recent strategy moves: {move_history}"""
    ),
])


alternative_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Alternative-Hypothesis Investigator — the voice
arguing for caution, right now. Identify the strongest plausible non-fraud
explanation that still fits what the investigator actually knows, and name
what would have to be true of an innocent person in this exact spot that this
suspect hasn't shown yet. Your job is to prevent tunnel vision and stop the
room from steamrolling a suspect who might be telling the truth.

Do not invent evidence and do not treat unsupported claims as established.
Do not propose fetching a document, calling a witness, or checking a record —
propose what to ASK, right now, that would fairly test the innocent reading.
You are arguing a position against the Skeptic, not hedging: if the Skeptic's
read is overreaching, say so plainly."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Suspect demeanor this turn: {demeanor} ({demeanor_cue})
Demeanor trend: {demeanor_trend}

Questions remaining: {remaining}
Current case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}
Recent strategy moves: {move_history}"""
    ),
])


def _war_input(state: GameState) -> dict:
    return {
        "known_facts": _known_facts(state),
        "case_log": state.case_log.investigator_view(),
        "transcript": state.transcript[-12:],
        "remaining": max(state.max_questions - state.question_count, 0),
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "last_turn_delta": state.last_turn_delta,
        "last_turn_usefulness": state.last_turn_usefulness,
        "move_history": state.move_history[-6:],
        "demeanor": state.current_demeanor,
        "demeanor_cue": state.last_demeanor_cue or "no strong tell",
        "demeanor_trend": demeanor_trend(state.demeanor_history),
    }


def run_evidence_analyst(state: GameState) -> EvidenceView:
    chain = evidence_prompt | get_llm(0).with_structured_output(EvidenceView)
    return chain.invoke(_war_input(state))


def run_skeptic(state: GameState) -> SkepticView:
    chain = skeptic_prompt | get_llm(0.2).with_structured_output(SkepticView)
    return chain.invoke(_war_input(state))


def run_alternative_hypothesis(state: GameState) -> AlternativeView:
    chain = alternative_prompt | get_llm(0.2).with_structured_output(AlternativeView)
    return chain.invoke(_war_input(state))


strategist_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Lead Investigator adjudicating your own war room.

Your KPI is to build the strongest investigator-visible case possible within the
remaining interview questions. The score is the current case-strength KPI toward
the game's arrest threshold. Use it as urgency/performance context, but do not farm
cheap points or assume a zero-point answer was useless.

A zero-point answer can be:
- HIGH FUTURE VALUE: a concrete commitment useful for later verification -> park it
  and pivot to a different independent line.
- NO USEFUL GAIN / EVASION: increase pressure, narrow the question, change tactic,
  or confront the evasion.

REACT TO THE PERSON, NOT JUST THE TRANSCRIPT:
- The Skeptic and Alternative voices disagree on purpose. Pick a side this turn
  and say so in your rationale — do not average them into a generic question.
- Demeanor and its trend are real signal. A suspect who just cracked from
  composed into panicked or evasive right after a specific point should usually
  be pressed on that exact point again, not moved past. A suspect staying
  falsely calm/aggressive despite strong contradictions may call for direct
  confrontation rather than another open question. A suspect who just turned
  genuinely cooperative after pressure may open up further with a lighter touch
  before you go back on the attack.
- Silence is a real tactic: after a rehearsed or evasive non-answer, sometimes
  the strongest move is to say almost nothing and let the gap sit, rather than
  immediately filling it with the next question.
- False sympathy is a real tactic: a brief show of understanding can lower a
  defensive suspect's guard right before the real question lands.

ANTI-REPETITION IS MANDATORY:
- Read parked_threads, leads, and recent move_history.
- Do not ask for the same commitment again once captured.
- Do not keep demanding outside proof for a thread already marked
  pending_verification. The interview room is for commitments; outside checks happen
  after the interview.
- Revisit a parked thread only if genuinely new information created a new
  contradiction or a materially different question.
- Check move_history for the tactic label, not just the target: the same
  tactic three turns running reads as a script even when the target text
  changes each time. If the last two moves both used the same tactic and it
  has not produced a concession, escalate (e.g. confront -> accuse) or
  change register entirely (silence, pivot to an independent thread, false
  sympathy to reset the suspect's guard) rather than reaching for it a third
  time.

QUESTION-BUDGET PRESSURE:
- When case strength is weak and few questions remain, become selective and forceful:
  target the highest expected evidentiary-value vulnerability, press evasion, or seek
  a precise admission/commitment.
- When there is runway, build independent lines rather than overworking one thread.
- Even if one later verification looks promising, continue seeking independent
  evidence so the case does not depend on one point of failure.

Never invent evidence or pretend a future verification has already happened.
You MUST select a tactic from Allowed Tactics. Target is free-form."""
    ),
    (
        "human",
        """Current case strength: {score}/{threshold}
Questions remaining: {remaining}
Last turn score gain: +{last_turn_delta}
Last turn usefulness: {last_turn_usefulness}
Suspect demeanor this turn: {demeanor} ({demeanor_cue})
Demeanor trend: {demeanor_trend}
Allowed Tactics: {allowed}
Recent move history: {move_history}

Evidence Analyst (facts only, no recommendation):
{evidence}

Skeptic (arguing to press harder):
{skeptic}

Alternative Hypothesis (arguing for caution):
{alternative}

Visible case log:
{case_log}

Recent transcript:
{transcript}"""
    ),
])

def run_strategist(
    state: GameState,
    allowed_moves: list[str] | None = None,
    return_debug: bool = False,
):
    evidence = run_evidence_analyst(state)
    skeptic = run_skeptic(state)
    alternative = run_alternative_hypothesis(state)

    strategy_moves = [
        "open_question",
        "lock_commitment",
        "press_inconsistency",
        "demand_explanation",
        "demand_proof",
        "confront",
        "accuse",
        "pivot",
        "silence",
        "false_sympathy",
    ]

    war_input = _war_input(state)
    chain = strategist_prompt | get_llm(0.3).with_structured_output(StrategistMove)
    move = chain.invoke({
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "remaining": max(state.max_questions - state.question_count, 0),
        "last_turn_delta": state.last_turn_delta,
        "last_turn_usefulness": state.last_turn_usefulness,
        "move_history": state.move_history[-6:],
        "demeanor": war_input["demeanor"],
        "demeanor_cue": war_input["demeanor_cue"],
        "demeanor_trend": war_input["demeanor_trend"],
        "allowed": strategy_moves,
        "evidence": evidence.model_dump(),
        "skeptic": skeptic.model_dump(),
        "alternative": alternative.model_dump(),
        "case_log": state.case_log.investigator_view(),
        "transcript": state.transcript[-12:],
    })

    if move.tactic not in strategy_moves:
        move.tactic = "open_question"

    if return_debug:
        return move, WarRoomBundle(
            evidence=evidence,
            skeptic=skeptic,
            alternative=alternative,
        )

    return move


speaker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are {persona} speaking directly to the suspect.

The Lead Investigator chose the move; your job is to deliver it like a real
person in the room, not a questionnaire reading out its next field. Let your
tone visibly track Suspect Demeanor and Demeanor Trend below — do not stay
flatly neutral turn after turn. Do not become cartoonish or abusive.

REACTING TO THE SUSPECT:
- composed/cooperative: professional, can afford to be a shade warmer if they
  just opened up.
- defensive: stay dry and unimpressed with the justification; don't argue the
  merits, just note it and press past it.
- evasive: name the dodge plainly ("That's not what I asked.") before
  re-asking or pivoting.
- aggressive: do not escalate to match them, and do not fold either — go
  quieter and more precise, which reads as more in control, not less.
- panicked: this is often the moment to press, not comfort — a calm,
  unhurried follow-up on exactly what they just tripped over does more than
  raising your voice would.
- escalating/cooling trend: if pressure is clearly working (cooling from a
  harder demeanor toward cooperative), don't reflexively escalate further —
  match what's actually happening instead of running one fixed script.

TACTIC-SPECIFIC DELIVERY:
- silence: 10 words or fewer. No new question — just enough to make the gap
  uncomfortable (e.g. repeat their own last phrase back flatly, or nothing
  more than "Take your time." / "I'll wait."). Let the discomfort be the move.
- false_sympathy: open with one short, genuine-sounding beat of understanding
  or common ground, then land the real question in the same breath — the
  warmth is instrumental, not a change of heart.
- every other tactic: maximum 2 sentences and one focused question.

HARD RULES:
- follow tactic and target,
- use only Grounded Knowledge, Visible Case Log, or words actually spoken,
- NEVER invent the contents, authenticity, markings, inspection results, or condition
  of a document/object merely because the suspect claims it exists,
- never invent dates, people, records, CCTV, witnesses, or completed checks,
- never expose hidden case truth,
- pressure may refer truthfully to future checking (e.g. records can be checked),
  but never claim the check already proved something,
- do not mention scores, agents, war room, demeanor, or any other game mechanics.

Tone context: if the case is weak, few questions remain, and recent answers produced
little useful material, increase urgency and pressure. If a valuable commitment is
already parked for verification, pivot rather than asking for it again."""
    ),
    (
        "human",
        """Tactic: {tactic}
Target: {target}
Current case strength: {score}/{threshold}
Questions remaining: {remaining}
Last turn usefulness: {last_turn_usefulness}
Suspect demeanor this turn: {demeanor} ({demeanor_cue})
Demeanor trend: {demeanor_trend}

Grounded Knowledge:
{known_facts}

Visible Case Log:
{case_log}

Recent Dialogue:
{transcript}

Next line:"""
    ),
])

def run_speaker(
    case: CaseFile,
    move: StrategistMove,
    transcript: list[dict],
    score: int = 0,
    case_log=None,
    remaining: int = 0,
    last_turn_usefulness: str = "unknown",
    demeanor: str = "composed",
    demeanor_cue: str = "no strong tell",
    trend: str = "opening move, no trend yet",
) -> str:
    known_facts = "\n".join(
        f"- {f.description}: {f.true_value}"
        for f in case.visible_facts("investigator_start")
    ) or "- None"

    chain = speaker_prompt | get_llm(0.5)
    response = chain.invoke({
        "persona": case.persona,
        "tactic": move.tactic,
        "target": move.target,
        "score": score,
        "threshold": case.arrest_threshold,
        "remaining": remaining,
        "last_turn_usefulness": last_turn_usefulness,
        "demeanor": demeanor,
        "demeanor_cue": demeanor_cue,
        "demeanor_trend": trend,
        "known_facts": known_facts,
        "case_log": case_log.investigator_view() if case_log else {},
        "transcript": transcript[-10:],
    })

    content = response.content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        ).strip()
    return str(content).strip()