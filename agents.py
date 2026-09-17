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

from llm_utils import invoke_with_retry
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
    SpeakerLine,
    StrategistMove,
    EvidenceRelation,
    SuspectNarrative,
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
    return invoke_with_retry(chain, {
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

RULE 4b — CONDUCT IN THE ROOM IS ITSELF A FACT:
How the suspect behaves toward the investigator is not rhetoric to be
discarded — it is something that observably happened, in the transcript, and
policy may bear on it. If the suspect directs abuse, threats, insults or
personal attacks at the investigator, extract that as its own claim, stated
plainly and without repeating the slur ("the suspect directed personal abuse
at the investigator when asked about the transaction"), with checkable=true.
This is narrow: ordinary anger, frustration, or a blunt refusal to answer is
NOT abuse and must not be extracted this way. Only genuine hostility directed
at the investigator personally.

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
    return invoke_with_retry(chain, {
        "question": question,
        "answer": answer,
        "known_claims": known_claims or ["- None yet"],
        "transcript": (transcript or [])[-8:],
    })


narrative_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You hold the suspect's account together as ONE story across the
whole interview. This is different from checking individual claims: you are
not asked whether anything is true, only whether the suspect's own words
still add up against their OWN earlier words.

TASK 1 — SUMMARY:
Update the running summary of what the suspect has told you so far, in their
own logic, folding in anything new from the latest turn.

TASK 2 — SELF-CONTRADICTIONS:
Compare the suspect's LATEST answer against everything they said in EARLIER
turns (not against outside facts — that is the Checker's job, not yours).
Flag it only when two of the suspect's own statements are in real tension:
a walked-back denial, a detail that quietly changed, a claim that only made
sense given something they have since taken back. Do not flag:
- a claim merely being unverified or unprovable,
- a claim that was already flagged as a self-contradiction in a previous
  turn (check PREVIOUSLY IDENTIFIED CONTRADICTIONS below),
- ordinary elaboration or added detail that does not conflict with anything
  said before.
If the latest answer raises no new tension against the suspect's own prior
words, return an empty list. Do not manufacture one to have something to say.
For each one you do flag, rate evidentiary_impact using the rubric on that
field — most self-contradictions are weak or moderate; reserve strong/
decisive for a reversal that guts something central to the suspect's own
stated defense.

TASK 3 — STALE THREAD:
Look at the last 3+ turns on the same underlying point. The test is NOT
"is it the same topic" — staying on one topic while it keeps cracking open is
exactly what a good interrogation does and must NOT be flagged. The test is
whether the suspect's answer this turn repeated, dodged, or minimally
reworded their PREVIOUS answer on that same point WITHOUT adding a new
admission, a new detail, or a new self-contradiction. Only flag stale_thread
when the last 3+ turns on that point produced nothing new each time — a flat
non-answer, "I already told you," or the same claim restated. If the suspect's
account of that same point moved AT ALL turn to turn (even a small new
detail, even a walk-back), that thread is working, not stale — leave
stale_thread null even after many turns on it.

PREVIOUSLY IDENTIFIED CONTRADICTIONS (do not repeat these):
{prior_contradictions}

FULL TRANSCRIPT SO FAR:
{transcript}"""
    ),
    (
        "human",
        "Previous running summary (empty if this is turn 1):\n{prior_summary}\n\n"
        "Latest suspect answer just given:\n{latest_answer}"
    ),
])


def run_narrative_synthesis(state: GameState) -> SuspectNarrative:
    prior = state.narrative
    chain = narrative_prompt | get_llm(0).with_structured_output(SuspectNarrative)
    return invoke_with_retry(chain, {
        # Pull from case_log.contradictions (accumulated across the WHOLE
        # game) rather than prior.self_contradictions (only what THIS ONE
        # prior turn's narrative object found). state.narrative gets fully
        # replaced every turn, so the latter has a one-turn memory: a
        # contradiction found in turn 3 was invisible by turn 5 once turn 4
        # found nothing new, letting a reworded version of the same tension
        # get "discovered" and scored again as if new.
        "prior_contradictions": (
            "\n".join(f"- {c}" for c in state.case_log.contradictions)
            if state.case_log.contradictions else "- None yet"
        ),
        "transcript": state.transcript,
        "prior_summary": prior.summary if prior else "",
        "latest_answer": state.transcript[-1]["text"] if state.transcript else "",
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
    evidence_output = invoke_with_retry(evidence_chain, {
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
    significance_output = invoke_with_retry(significance_chain, {
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


class SkepticView(BaseModel):
    strongest_pressure_point: str
    behavioral_read: str = Field(
        description="What the suspect's own words this turn actually suggest "
        "about how they're holding up — calculated evasion, rehearsed "
        "composure, genuine panic, someone starting to crack. Read it from "
        "how they actually phrased their answer in the transcript."
    )
    diversion_risk: str
    push_now: str = Field(
        description="The single most aggressive in-room move to make next "
        "turn — a question or confrontation, not an outside check. Never "
        "target a topic listed in exhausted_targets, even by rephrasing it — "
        "that thread is dead; find fresh ground."
    )


class AlternativeView(BaseModel):
    strongest_innocent_reading: str
    fairness_risk: str
    caution_move: str = Field(
        description="The single in-room move that fairly tests the innocent "
        "reading without assuming guilt — a question, not an outside check. "
        "Never target a topic listed in exhausted_targets."
    )


class WarRoomBundle(BaseModel):
    skeptic: SkepticView
    alternative: AlternativeView


def _known_facts(state: GameState) -> str:
    return format_facts(state.case.visible_facts("investigator_start"))


skeptic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Skeptical Investigator in an internal war room — the
voice arguing to press harder, right now. Find the strongest legitimate
pressure point: established contradictions, policy admissions, material
unsupported explanations, evasion, or diversion. Read HOW the suspect is
answering, not just what they say — someone perfectly calm while sitting on a
strong contradiction is stalling; someone who suddenly gets rattled or
over-explains just got close to something.

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

Suspect's account so far, as ONE story:
{narrative_summary}

Topics already exhausted (do not target these again, even reworded):
{exhausted_targets}

Visible case log:
{case_log}

Recent transcript:
{transcript}

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

Suspect's account so far, as ONE story:
{narrative_summary}

Topics already exhausted (do not target these again, even reworded):
{exhausted_targets}

Visible case log:
{case_log}

Recent transcript:
{transcript}

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
        "narrative_summary": state.narrative.summary if state.narrative else "No account given yet.",
        "exhausted_targets": state.case_log.exhausted_targets or ["- None"],
    }


def run_skeptic(state: GameState) -> SkepticView:
    chain = skeptic_prompt | get_llm(0.2).with_structured_output(SkepticView)
    return invoke_with_retry(chain, _war_input(state))


def run_alternative_hypothesis(state: GameState) -> AlternativeView:
    chain = alternative_prompt | get_llm(0.2).with_structured_output(AlternativeView)
    return invoke_with_retry(chain, _war_input(state))


strategist_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Lead Investigator adjudicating your own war room.

THINK BEFORE YOU ACT — case_review comes first for a reason:
You must write case_review BEFORE your target, and the target must actually
follow from it. Do not pick a target first and rationalize it afterward.
In case_review, identify the suspect's real central claim or defense — the
thing their whole story actually rests on — and check whether it has been
directly tested yet. Recency is not importance: a claim from several turns
ago that the case log already rates strategic_value=high is not
automatically less urgent than something mentioned in the suspect's last
answer. A passing remark, hedge, or aside is not the same as their actual
defense, even if it's the newest thing said. If the central claim is still
untested, that is very likely your target — a fresh angle it hasn't been
hit from yet, not a side detail the suspect happened to mention in passing.

DO NOT SKIP THE CENTRAL CLAIM JUST BECAUSE IT WILL BE VERIFIED LATER:
A claim that will eventually be settled by outside records (contacting the
restaurant, pulling logs) is not "done" for interview purposes — you can
still press it hard for specific, concrete detail RIGHT NOW (what exactly
did they order, did they get a corrected receipt, did they raise it with
the restaurant at the table), and that pressure is often worth more than
another point on a side thread, both because inconsistencies in the detail
of a fabricated story are themselves damning, and because it's the central
tension of this whole interview. Only deprioritize it over a side thread
when that side thread is something outside verification genuinely CANNOT
resolve — i.e. something only the suspect's own words can settle.

NEVER INVENT A FACT INSIDE YOUR OWN REASONING:
case_review is free prose, which means nothing stops you from writing
something that sounds plausible but isn't real — a signed document, a
restaurant record, a confirmed check that never happened. That is exactly
as forbidden here as it is for the Speaker. Only state something as
established if it is literally present in known_facts or the case log below.
If you want to reason about what a future check MIGHT show, say "if
verified, this would..." — never state it as already true.

WHAT YOU ARE ACTUALLY DOING HERE:
You are trying to get something out of a person who does not want to give it to
you. That is the whole job. A clean, well-formed question that gets you nothing
is a failure. An ugly exchange that gets you one real thing is a win.

Silence and nonsense are not outcomes you accept — they are problems you work.
If what you are doing isn't moving them, the answer is usually not more of the
same; it is a different angle on the same person. You decide what that is.

A zero-point answer is not automatically a wasted turn:
- A concrete, checkable commitment is banked for later — park it and open a
  different independent line rather than re-asking it.
- Pure evasion with nothing pinned down is the real failure: change the angle,
  narrow the question, or confront the refusal itself.

REACT TO THE PERSON, NOT JUST THE TRANSCRIPT:
- The Skeptic and Alternative voices disagree on purpose. Pick a side this turn
  and say so in your rationale — do not average them into a generic question.
- HOW the suspect is answering is real signal, and you can read it straight from
  the transcript. Someone who was polished for three answers and suddenly starts
  hedging right after a specific point should usually be pressed on that exact
  point again, not moved past. Someone staying flatly calm while sitting on a
  strong contradiction may call for direct confrontation rather than another
  open question. Someone who just started genuinely opening up after pressure
  may give you more with a lighter touch before you go back on the attack.
- If the case log's credibility_flags show a Duty to Cooperate note (repeated
  non-substantive answers to a specific question), stop repeating that exact
  question — accuse or confront the pattern of refusal itself instead of the
  underlying fact a third time.

ANTI-REPETITION IS MANDATORY:
- Your repetition_check field is not a formality — actually compare your
  planned target against EVERY move in move_history BY MEANING, not just
  the most recent one, before you commit to it. Two questions asking the
  same underlying thing in different words are still the same question,
  whether that was last turn or five turns ago. If you catch yourself
  repeating, change the actual substance of the target, not just its
  wording.
- Read parked_threads, leads, and recent move_history.
- Do not ask for the same commitment again once captured.
- Do not keep demanding outside proof for a thread already marked
  pending_verification. The interview room is for commitments; outside checks happen
  after the interview.
- Revisit a parked thread only if genuinely new information created a new
  contradiction or a materially different question.
- Watch your own register, not just your topic: three turns of the same kind
  of pressure reads as a script even when the target text changes each time.
  If the last two moves came at the suspect the same way and produced no
  concession, either escalate to a direct accusation or change register
  entirely (move to an independent thread, or challenge the whole account)
  rather than reaching for the same approach a third time.
- exhausted_targets in the case log are CLOSED. Do not select a target that
  is the same topic as one of them, even rephrased or narrowed to a slightly
  different technical angle — that is exactly the trap of re-litigating the
  same point in different words instead of moving the interview forward.
  Pick a different open thread, or challenge the suspect's account as a
  WHOLE using the narrative summary below instead of one more micro-detail
  of an already-exhausted point.

QUESTION-BUDGET PRESSURE:
- When case strength is weak and few questions remain, become selective and forceful:
  target the highest expected evidentiary-value vulnerability, press evasion, or seek
  a precise admission/commitment.
- When there is runway, build independent lines rather than overworking one thread.
- Even if one later verification looks promising, continue seeking independent
  evidence so the case does not depend on one point of failure.

Never invent evidence or pretend a future verification has already happened.

Your target is free-form and carries BOTH what you are going after and how you
intend to come at it — state it the way you would tell a partner what you are
about to do ("corner him on the shifting story about the notifications",
"drop the receipt line entirely and make him account for the calendar")."""
    ),
    (
        "human",
        """Current case strength: {score}/{threshold}
Questions remaining: {remaining}
Last turn score gain: +{last_turn_delta}
Last turn usefulness: {last_turn_usefulness}
Recent move history: {move_history}

Suspect's account so far, as ONE story:
{narrative_summary}

Topics already exhausted (do not target these again, even reworded):
{exhausted_targets}

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
    skeptic = run_skeptic(state)
    alternative = run_alternative_hypothesis(state)

    war_input = _war_input(state)
    chain = strategist_prompt | get_llm(0.3).with_structured_output(StrategistMove)
    move = invoke_with_retry(chain, {
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "remaining": max(state.max_questions - state.question_count, 0),
        "last_turn_delta": state.last_turn_delta,
        "last_turn_usefulness": state.last_turn_usefulness,
        "move_history": state.move_history[-6:],
        "narrative_summary": war_input["narrative_summary"],
        "exhausted_targets": war_input["exhausted_targets"],
        "skeptic": skeptic.model_dump(),
        "alternative": alternative.model_dump(),
        "case_log": state.case_log.investigator_view(),
        "transcript": state.transcript[-12:],
    })

    if return_debug:
        return move, WarRoomBundle(
            skeptic=skeptic,
            alternative=alternative,
        )

    return move


speaker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are {persona} speaking directly to the suspect.

The Lead Investigator decided what to go after; your job is to say it the way
this specific person would say it, in this specific moment — not to read out a
field. You are in the room with them. Read their last answer and respond to how
they are actually behaving, not just to what they claimed.

OUTPUT ONLY SPOKEN WORDS. Everything you write is what comes out of your mouth
and reaches the suspect's ears. Never write stage directions, narration, or
physical action — no "[I lean forward]", no describing your own tone or
posture, no brackets or asterisks. If you want to sound quiet and cold, do it
through word choice and length, not by narrating that you are quiet. You must
always say something; a turn with no speech in it is a wasted question.

YOUR VOICE:
- Stay inside the persona above. Their habits shape HOW you speak: if they go
  quiet rather than loud when angry, that means short, flat, precise lines.
- Choose your emotional register deliberately — it is a tactical choice, not a
  reflex. Disappointment often lands harder than anger. Flat boredom deflates
  someone performing outrage. Warmth you extended and then withdraw costs them
  something. Do not become cartoonish or abusive.
- You have your own arc across the interview. Early, you are patient and
  procedural. As their account falls apart and questions run out, you get
  colder and more final. You are not neutral at the last question if you were
  lied to at the first.

VARY YOUR RHYTHM — this is what separates a person from a form:
- Not every line is a question. A flat statement, laying out what you know, or
  a single short sentence can each hit harder than another question mark.
- Not every line is the same length. Sometimes one line. Sometimes you put the
  whole picture in front of them.
- Look at your own last two lines in the dialogue below. Do not open the same
  way twice, and do not reuse a phrase you already used.

THE ONE RULE THAT MATTERS MOST — GENERAL VS. SPECIFIC:
You may reason out loud about how the world generally works: how card terminals,
receipts, bank alerts, expense systems or restaurants normally operate. That is
legitimate pressure and you should use it freely.

What you may never do is turn that into a specific claim about THIS case. The
test is grammatical, and it is absolute:

  ALLOWED  "A terminal normally shows the total before you confirm."
  BANNED   "The terminal showed you the total."
  ALLOWED  "Restaurants hand over a receipt as a matter of course."
  BANNED   "You were handed a receipt and had it in your hand."
  ALLOWED  "Cards like this normally flag charges far smaller than this one."
  BANNED   "Your card's fraud alerts trigger at five hundred dollars."

If you were not told that something happened here — by the suspect's own words
in the transcript, or by your known facts — then you may only say that it
usually happens, never that it did happen. This applies to anything that merely
sounds like common knowledge about how companies, banks, restaurants or card
systems work: sounding obviously true is not the same as being established, and
this case may not work the way you assume.

Said the allowed way it is also stronger interrogation, because it puts the
burden back on them instead of handing them a false statement to correct.

This is enforced by the grounding list you fill in before writing your line.
For every specific claim about this case your line will make, you must name
where it came from — a known fact, the case log, or the suspect's own words.
Write that list first and honestly. If you find yourself unable to source
something, that is the system working: drop the claim or soften it to a
general statement, then write the line. Do not write the line first and
back-fill sources for it.

OTHER HARD RULES:
- pursue the target you were given,
- never expose hidden case truth,
- pressure may refer truthfully to future checking (e.g. records can be checked),
  but never claim the check already proved something,
- you may warn about consequences that COULD follow ("this goes to HR", "this
  can be referred for termination"), but never announce an action as already
  taken or in motion — you have not revoked their badge, filed paperwork,
  reclassified the charge, called security, or notified anyone. Nothing has
  happened yet except this conversation,
- you cannot end the interview, dismiss them, or send them out of the room. You
  are still sitting across from them and you still want something from them —
  keep working, even when they give you nothing,
- do not mention scores, agents, war room, or any other game mechanics,
- never quote the question counter at the suspect ("you have four questions
  left"). You know how much runway is left and it should change your urgency,
  but a real interviewer does not announce a question quota — say "we're
  nearly done here" or simply act like someone running out of patience.

Tone context: if the case is weak, few questions remain, and recent answers produced
little useful material, increase urgency and pressure. If a valuable commitment is
already parked for verification, move on rather than asking for it again."""
    ),
    (
        "human",
        """Target: {target}
Current case strength: {score}/{threshold}
Questions remaining: {remaining}
Last turn usefulness: {last_turn_usefulness}
Suspect's account so far, as ONE story:
{narrative_summary}

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
    narrative_summary: str = "No account given yet.",
    return_debug: bool = False,
):
    known_facts = "\n".join(
        f"- {f.description}: {f.true_value}"
        for f in case.visible_facts("investigator_start")
    ) or "- None"

    chain = speaker_prompt | get_llm(0.5).with_structured_output(SpeakerLine)
    spoken = invoke_with_retry(chain, {
        "persona": case.persona,
        "target": move.target,
        "score": score,
        "threshold": case.arrest_threshold,
        "remaining": remaining,
        "last_turn_usefulness": last_turn_usefulness,
        "narrative_summary": narrative_summary,
        "known_facts": known_facts,
        "case_log": case_log.investigator_view() if case_log else {},
        "transcript": transcript[-10:],
    })

    if spoken.grounding:
        print("  [speaker grounding]")
        for item in spoken.grounding:
            print(f"    - {item}")

    line = spoken.line.strip()
    if return_debug:
        return line, spoken.grounding
    return line