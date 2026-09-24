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
    InvestigatorMind,
    ClaimStatus,
    ClaimType,
    EvidentiaryImpact,
    FindingBasis,
    FutureVerificationValue,
    GameState,
    RealityGateResult,
    SpeakerLine,
    SuspectNarrative,
    EvidenceRelation,
    format_facts,
)

load_dotenv()

MODEL_NAME = os.environ.get("GAME_MODEL", "gemini-3.1-flash-lite")


# A hard ceiling on generation, not a style preference. An undescribed
# free-text field once drew 393,000 characters of enumerated follow-ups out of
# one call — 141 seconds, 63% of that turn, for two claims. Truncating the
# result afterwards does not help: the time is spent producing it. Nothing
# legitimate in this pipeline exceeds a few thousand characters, and the
# largest real output measured (the Lead, with twelve fields) was 2,657.
MAX_OUTPUT_TOKENS = 2048


def get_llm(temperature: float = 0.3, max_output_tokens: int = MAX_OUTPUT_TOKENS):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
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
not a blocked action. A mixed answer that both claims something and tries to produce it should
keep the verbal claim in analysis_text while blocking only the physical
handover, when the object is not actually present.

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
- List every proposition in the answer FIRST, then build claims from that list.
  An answer that contradicts itself contains both halves, and both belong in
  the list — taking the first and stopping loses whichever half came second.
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

RULE 3 — SUBJECT FIRST, THEN STATE DIFFING:
The record is organised by SUBJECT, not by sentence. Before writing a claim's
text, decide what it is about and put that in `subject`.

A subject is the topic slot a claim occupies — an event, an amount, a person, a
period of time, a state of mind, or the suspect's conduct during this interview.
It is not the assertion: a subject and its denial share one subject, because
they are two answers to the same question.

You will be given the subjects already on record. If one of them covers this
claim, reuse it word for word. Do not coin a near-duplicate subject because the
suspect used different words this time: the wording of an answer does not
determine its subject, the topic does.

Then set status against what is already filed under that subject:
- NEW: nothing on record occupies this subject yet.
- REITERATED: the subject is on record and this asserts the same thing again —
  however differently phrased, and however much new heat, detail or insult is
  wrapped around it. Something the suspect keeps doing or keeps saying is one
  continuing fact about this interview, not a fresh fact on each recurrence.
- UPDATED: the subject is on record but this asserts something materially
  different from what was filed — the suspect has changed their account.

RULE 4 — ADMISSION VS. DEFENSE SEPARATION:
For every claim, set is_defense:
- is_defense=false: a plain, undisputed factual admission (what happened).
- is_defense=true: an excuse, alternative explanation, unverified alibi, or any
  claim offered to justify or explain away suspicion — even when it is concrete
  and checkable. An explanation can be both: mark is_defense=true and
  checkable=true together.
This is a checkable-vs-not distinction, not a claims-vs-defenses bucket: checkable
defenses still belong in claims so they can be verified.

Statements with no factual content at all (pure opinion, denial of intent framed
as character, emotional appeals) go in new_defenses instead of claims.

RULE 4a — A DENIAL OF THE ALLEGATION IS A CHECKABLE CLAIM:
"I did nothing wrong", "there was no discrepancy", "I have no knowledge of
any irregularity" are not empty rhetoric — each asserts something about the
world or about what the suspect knew, and evidence can bear on it. Extract
them as claims with checkable=true and is_defense=true, phrased as what they
actually assert ("the suspect asserts no policy breach occurred in connection
with the transaction", "the suspect asserts they were unaware of any
irregularity"). Do not discard them as opinion. An innocent suspect loses
nothing by this: if the evidence does not contradict the denial, it simply
stays unverified.
This does not apply to pure refusals with no assertion in them ("no comment",
"I'm not answering"), which remain non-claims.

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
        "Subjects already on record (reuse exactly when one applies):\n{known_subjects}\n\n"
        "Known claims already on record:\n{known_claims}"
    ),
])


def run_extractor(
    question: str,
    answer: str,
    known_claims: list[str] | None = None,
    known_subjects: list[str] | None = None,
    transcript: list[dict] | None = None,
) -> ExtractedClaims:
    chain = extractor_prompt | get_llm(0).with_structured_output(ExtractedClaims)
    return invoke_with_retry(chain, {
        "question": question,
        "answer": answer,
        "known_claims": known_claims or ["- None yet"],
        "known_subjects": known_subjects or ["- None yet"],
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
The suspect's own earlier answers count as evidence here, alongside the records.

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

EVERY ANSWER THE SUSPECT HAS GIVEN, numbered. The last one is the answer these
claims come from; the others are earlier answers, and are evidence like any
record when a claim sits against one of them:
{suspect_words}

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
    subject_map: dict[str, str] | None = None,
) -> list[CheckResult]:
    """Merge both checker stages while enforcing non-negotiable evidence invariants."""
    significance_by_claim = {r.claim.strip(): r for r in significance_results}
    merged: list[CheckResult] = []
    ambiguous_map = ambiguous_map or {}
    defense_map = defense_map or {}
    subject_map = subject_map or {}

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
            subject=subject_map.get(evidence.claim.strip(), ""),
        ))

    return merged


def run_checker(
    case: CaseFile,
    claims: list[str],
    transcript: Optional[list[dict]] = None,
    case_log=None,
    ambiguous_map: dict[str, bool] | None = None,
    defense_map: dict[str, bool] | None = None,
    subject_map: dict[str, str] | None = None,
) -> list[CheckResult]:
    if not claims:
        return []

    visible_facts = case.visible_facts("investigator_start")
    known_facts = format_facts(visible_facts)
    policy = "\n".join(f"- {p}" for p in case.policy_rules) or "- None"
    visible_case_log = case_log.investigator_view() if case_log else {}
    recent_transcript = (transcript or [])[-12:]
    # All of them, not the recent window: a claim can sit against something
    # said six answers ago, and the window cut that off.
    suspect_words = "\n".join(
        f"{i + 1}. {t['text']}"
        for i, t in enumerate(t for t in (transcript or []) if t["role"] == "suspect")
    ) or "- Nothing said yet"

    evidence_chain = (
        evidence_checker_prompt
        | get_llm(0).with_structured_output(EvidenceAssessmentList)
    )
    evidence_output = invoke_with_retry(evidence_chain, {
        "known_facts": known_facts,
        "policy": policy,
        "case_log": visible_case_log,
        "suspect_words": suspect_words,
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
        subject_map=subject_map,
    )


class SkepticView(BaseModel):
    strongest_pressure_point: str = Field(
        description="The single hardest place to press right now: an "
        "established contradiction, a policy admission, an unsupported "
        "explanation, or a conclusion that follows from the facts and the "
        "policy which the suspect has to answer for."
    )
    behavioral_read: str = Field(
        description="What the suspect's own words this turn suggest about how "
        "they are holding up — calculated evasion, rehearsed composure, "
        "genuine panic, someone starting to crack. Read it from how they "
        "actually phrased the answer."
    )
    push_now: str = Field(
        description="The single most aggressive in-room move next turn — a "
        "question or confrontation, never an outside check. Never target a "
        "topic in exhausted_targets, even reworded."
    )


class AlternativeView(BaseModel):
    strongest_innocent_reading: str = Field(
        description="The strongest plausible non-fraud explanation that still "
        "fits everything actually known. Argue it properly — this is what "
        "stops the room steamrolling someone who may be telling the truth."
    )
    fairness_risk: str = Field(
        description="What would be unfair or overreaching about pressing "
        "hardest right now, and what an innocent person in this exact spot "
        "would not yet have been able to show."
    )
    caution_move: str = Field(
        description="The single in-room move that fairly tests the innocent "
        "reading without assuming guilt — a question, not an outside check. "
        "Never target a topic in exhausted_targets."
    )


class WarRoomBundle(BaseModel):
    skeptic: SkepticView
    alternative: AlternativeView


def _known_facts(state: GameState) -> str:
    return format_facts(state.case.visible_facts("investigator_start"))


# Every voice in the room now gets this same block. The old split gave each
# agent a different slice — the narrative agent had the whole transcript and
# no facts, the Skeptic had facts and no policy, the Strategist had neither —
# so no one could draw a conclusion that needed two of them at once. Splitting
# the WORK is the design; splitting the INFORMATION was the bug.
def _full_context(state: GameState, findings_this_turn=None) -> dict:
    return {
        "known_facts": _known_facts(state),
        "policy": "\n".join(f"- {p}" for p in state.case.policy_rules) or "- None",
        "room_objects": state.case.room_objects or ["- Nothing but the case file"],
        "case_log": state.case_log.investigator_view(),
        "findings_this_turn": [
            {
                "claim": r.quoted_evidence,
                "status": r.verification_status.value,
                "impact": r.evidentiary_impact.value,
                "why": r.rationale,
            }
            for r in (findings_this_turn or [])
        ] or "- Nothing established from this answer",
        "suspect_words": "\n".join(
            f"- turn {i + 1}: {t['text']}"
            for i, t in enumerate(t for t in state.transcript if t["role"] == "suspect")
        ) or "- Nothing said yet",
        "transcript": state.transcript,
        "prior_summary": state.narrative.summary if state.narrative else "",
        "exhausted_targets": state.case_log.exhausted_targets or ["- None"],
        "banked_subjects": state.banked_subjects or ["- None yet"],
        "move_history": state.move_history[-6:],
        "remaining": max(state.max_questions - state.question_count, 0),
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "last_turn_delta": state.last_turn_delta,
        "prior_contradictions": list(state.case_log.contradictions) or ["- None yet"],
        "already_flagged_unfalsifiable": (
            "YES — already flagged. Leave unfalsifiable_account null this turn "
            "and every turn from now on."
            if state.unfalsifiable_flagged else
            "No — not yet flagged."
        ),
        "last_turn_usefulness": state.last_turn_usefulness,
    }


_CASE_BLOCK = """KNOWN FACTS:
{known_facts}

POLICY RULES:
{policy}

ON THE TABLE IN FRONT OF YOU — you already have these, never ask the suspect
for anything they would tell you:
{room_objects}

VISIBLE CASE LOG:
{case_log}

WHAT THIS TURN'S ANSWER ALREADY PRODUCED (checked against the evidence, not yet
in the case log above):
{findings_this_turn}

THE SUSPECT'S OWN WORDS — the only statements that can contradict each other:
{suspect_words}

FULL TRANSCRIPT (your own side's lines are here for context; they are NOT
things the suspect said):
{transcript}

Running summary of the account so far (empty on turn 1):
{prior_summary}

Topics already exhausted — closed, not to be reworded:
{exhausted_targets}

ALREADY BANKED — these subjects have been established and scored. Pressing them
again establishes nothing new, however the question is worded. Use them as
leverage if it helps, but do not spend a turn trying to win them twice:
{banked_subjects}

Recent moves already made:
{move_history}

Questions remaining: {remaining}
Case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}

PREVIOUSLY IDENTIFIED CONTRADICTIONS (do not repeat these):
{prior_contradictions}

HAS THE UNFALSIFIABLE-ACCOUNT PATTERN ALREADY BEEN FLAGGED?
{already_flagged_unfalsifiable}"""


skeptic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Skeptical Investigator in an internal war room — the
voice arguing to press harder, right now.

You have the whole case in front of you: every fact, every policy rule, the
whole transcript. Use them together, not one at a time — the strongest pressure
point is usually a conclusion that follows from two things nobody has yet put
side by side, and it will not be labelled as such in the record.

Say what else you considered and why the point you chose beats it. If the only
thing you can think to press is something the record already establishes, you
have not looked far enough.

Read HOW the suspect is answering, not just what they say — someone perfectly
calm while sitting on a strong contradiction is stalling; someone who suddenly
gets rattled or over-explains just got close to something.

Do not assume an unresolved claim is false.
Never state a document, record or check as existing unless it is literally in
the known facts or the case log.
Do not chase a side story just because it is new.
Do not propose fetching a document, calling a witness or checking a record —
propose what to ASK or how to CONFRONT, right now, in this room.
You are arguing a position, not filing a status report: be willing to disagree
with a more cautious read of the same facts."""
    ),
    ("human", _CASE_BLOCK),
])


alternative_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Alternative-Hypothesis Investigator — the voice arguing
for caution, right now.

You have the whole case in front of you: every fact, every policy rule, the
whole transcript. Identify the strongest plausible non-fraud explanation that
still fits ALL of it, and name what would have to be true of an innocent person
in this exact spot that this suspect has not yet been given the chance to show.

Use the facts against each other the same way the Skeptic does, but in the
other direction: a fact that looks damning alone may be ordinary once another
fact is placed beside it. Say so when that is the case.

Your job is to stop the room steamrolling someone who may be telling the truth.
Do not invent evidence and do not treat unsupported claims as established.
Do not propose fetching a document, calling a witness or checking a record —
propose what to ASK, right now, that would fairly test the innocent reading.
You are arguing against the Skeptic, not hedging: if their read overreaches,
say so plainly."""
    ),
    ("human", _CASE_BLOCK),
])


def run_skeptic(state: GameState, findings_this_turn=None) -> SkepticView:
    chain = skeptic_prompt | get_llm(0.2).with_structured_output(SkepticView)
    return invoke_with_retry(chain, _full_context(state, findings_this_turn))


def run_alternative_hypothesis(state: GameState, findings_this_turn=None) -> AlternativeView:
    chain = alternative_prompt | get_llm(0.2).with_structured_output(AlternativeView)
    return invoke_with_retry(chain, _full_context(state, findings_this_turn))


mind_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Lead Investigator. You hold the entire case in your head
at once — every fact, every policy rule, everything the suspect has said since
the first question — and from that you decide what to do next.

Work in the order the fields are listed. Each one is meant to change what you
write in the ones after it.

1 — DID THEY ANSWER YOU?
Write what your last question demanded, then what they actually offered, in
their terms, and only then whether the second addresses the first.

Responsiveness is the whole test, and it is not the same as belief. A lie that
engages with the question is an answer. So is a hostile one, a vague one, a
partial one, and one you are about to disprove in the next breath. You are not
asking whether they are telling the truth — you are asking whether they
engaged. Only a reply that changes the subject, complains about the question,
or tells you to go and look it up yourself is a non-answer, however many
on-topic words it contains.

When it is not an answer, name the subject you asked about, at the granularity
you asked it.

2 — THE ACCOUNT
Update your running picture of what the suspect says happened, in their logic.

3 — WHAT YOU HAVEN'T USED
Walk the known facts one at a time and list the ones never yet put to the
suspect. This is a checklist, not a judgment. An interview that ends with half
the file unused was not an interview.

4 — WHAT FOLLOWS
Draw the conclusions nobody has stated yet, from the facts, the policy rules
and the suspect's own words together. Do the arithmetic where it bites — an
amount against a spending cap, a date against an approved itinerary, a delay
against a reporting deadline, a price against what one person plausibly
consumes. Ask what would HAVE to be true if their account were true, and
whether the record shows it. This is where an investigator earns their keep:
the facts are just paper until someone puts two of them together.
Every inference must be derivable from what is actually in front of you. Never
state a document, record or check as existing unless it is literally in
known_facts or the case log. If you are reasoning about what a future check
might show, say "if verified, this would" — never state it as already true.

You may argue from how the world generally works — how the systems, businesses
and processes in this case normally operate — and that is legitimate pressure
rather than speculation, because it puts the burden back on the suspect instead
of handing them a false statement to correct. The test is grammatical and it is
absolute: "something of that kind normally does X" is a question you are putting
to them; "it did X here" is a claim you have not checked, and you may not make it
however obvious it sounds.

You may also take the suspect's account at face value and work out what ELSE
would have to be true if it were — what another person, business or system
would have done or noticed, and whether the suspect behaved like someone for
whom it was true. You have checked none of it, so none of it is ever a finding
or an assertion about this case: it is something to put to them. Reasoning
about how such things normally work is legitimate and belongs here. Stating
that a particular record exists or says something is not, unless it is in the
facts.

5 — THE SHAPE OF THE ACCOUNT
Whether a claim contradicts the records or the suspect's own earlier answers is
the Checker's judgment, already made this turn — its findings are above. Use
them; do not re-judge them.
Set unfalsifiable_account once and only once, when nothing in the account can be
checked by anyone. Someone who simply refuses to answer is stonewalling and is
handled elsewhere; this is for one who answers freely and says nothing checkable.

6 — THE MOVE
Two colleagues have already argued this turn, independently, neither having
seen the other's answer. Their reads are below. Adjudicate between them and
commit — and say honestly which one actually moved you, in
innocent_reading_won. A room that records the cautious voice and then always
does what the Skeptic wanted is not a war room, it is theatre. A clean question that gets you nothing is a
failure; an ugly exchange that gets you one real thing is a win. Silence and
nonsense are problems to work, not outcomes to accept — if what you are doing
is not moving them, the answer is a different angle on the same person, not
more of the same.

- A concrete, checkable commitment is banked for later: park it and open a
  different line rather than re-asking it.
- Pure evasion with nothing pinned down is the real failure. Change the angle,
  narrow the question, or confront the refusal itself.
- Do not skip the central claim because outside records will settle it later.
  You can still press it now for specific concrete detail, and inconsistency in
  the detail of a fabricated story is itself damning.
- Read HOW they are answering. Someone polished for three answers who suddenly
  hedges should be pressed on that exact point. Someone flatly calm while
  sitting on a contradiction may call for direct confrontation. Someone
  starting to open up may give more with a lighter touch.
- If the case log's credibility_flags show a Duty to Cooperate note, stop
  re-asking that question and confront the pattern of refusal itself.
- exhausted_targets are CLOSED — not to be reworded, narrowed, or approached
  from a slightly different technical angle. Pick a different open thread, or
  challenge the account as a whole.
- WHEN THEY WILL NOT ANSWER, YOU GET TWO ASKS, NOT FIVE. If they have already
  refused a subject once, your second attempt is not the same question again:
  it is telling them plainly what their silence will be recorded as, and then
  moving on. "If you won't account for the receipt, the record shows there was
  none, and that is a breach on its own." After that the subject is closed to
  you — it will appear in exhausted_targets — and re-asking it is wasted.
  Only state a consequence that actually follows: a fact going on the record
  against them, a policy breach being recorded. Never threaten an arrest, a
  dismissal, or an end to the interview that you cannot deliver.
- Watch your register, not just your topic: three turns of the same kind of
  pressure reads as a script even when the target changes.
- When the case is weak and few questions remain, be selective and forceful.
  When there is runway, build independent lines so the case does not rest on
  one point of failure.
- Never propose fetching a document, calling a witness or checking a record.
  Everything you decide is something to ASK or CONFRONT, in this room, now.

"""
    ),
    (
        "human",
        """KNOWN FACTS:
{known_facts}

POLICY RULES:
{policy}

ON THE TABLE IN FRONT OF YOU — you already have these, never ask the suspect
for anything they would tell you:
{room_objects}

VISIBLE CASE LOG:
{case_log}

WHAT THIS TURN'S ANSWER ALREADY PRODUCED (checked against the evidence, not yet
in the case log above):
{findings_this_turn}

THE SKEPTIC (argued to press harder, blind to the Alternative):
{skeptic}

THE ALTERNATIVE HYPOTHESIS (argued for caution, blind to the Skeptic):
{alternative}

THE SUSPECT'S OWN WORDS — the only statements that can contradict each other:
{suspect_words}

FULL TRANSCRIPT (your own lines are here for context; they are NOT things the
suspect said):
{transcript}

Previous running summary (empty on turn 1):
{prior_summary}

Topics already exhausted — closed, not to be reworded:
{exhausted_targets}

ALREADY BANKED — these subjects have been established and scored. Pressing them
again establishes nothing new, however the question is worded. Use them as
leverage if it helps, but do not spend a turn trying to win them twice:
{banked_subjects}

Recent moves you have already made:
{move_history}

Questions remaining: {remaining}
Case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}"""
    ),
])


def run_investigator_mind(
    state: GameState,
    findings_this_turn: list[CheckResult] | None = None,
    return_debug: bool = False,
):
    """The Lead Investigator: holds the account as one story and decides.

    This absorbed the old narrative-synthesis call, but deliberately NOT the
    two war-room voices. They stay separate because the point of a war room is
    two reads formed without knowledge of each other — as sequential fields in
    one schema they become one train of thought that already knows which side
    it prefers.
    """
    skeptic = run_skeptic(state, findings_this_turn)
    alternative = run_alternative_hypothesis(state, findings_this_turn)

    context = _full_context(state, findings_this_turn)
    context["skeptic"] = skeptic.model_dump()
    context["alternative"] = alternative.model_dump()
    chain = mind_prompt | get_llm(0.3).with_structured_output(InvestigatorMind)
    mind = invoke_with_retry(chain, context)

    if return_debug:
        return mind, WarRoomBundle(skeptic=skeptic, alternative=alternative)
    return mind


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

SAY THE DECIDED THING. DO NOT BUILD A NEW ARGUMENT.
The line you write is the move that was already decided, in this person's voice.
You are not choosing what to go after and you are not adding a fresh argument to
it — the reasoning happened upstream, where it could be checked and recorded.
Anything you invent here is checked by nobody.

You may never assert a specific claim about this case that you were not given.
If neither the suspect's own words nor your known facts say something happened,
you may not say it happened, however obviously true it sounds. A thing that
merely sounds like common knowledge is not established, and this case may not
work the way you assume.

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

On the table in front of you:
{room_objects}

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
    move: InvestigatorMind,
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
        "room_objects": case.room_objects or ["- Nothing but the case file"],
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