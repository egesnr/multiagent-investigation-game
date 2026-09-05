"""
Interview agents.

Flow:
Extractor -> Checker -> Shared Context -> War Room -> Lead Strategist -> Speaker

Knowledge boundary:
- Checker may see full authored truth.
- War Room sees only investigator-known facts + investigator-visible case log.
- Speaker sees only investigator-known facts + visible interview context.
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
    FindingBasis,
    GameState,
    StrategistMove,
)

load_dotenv()

MODEL_NAME = os.environ.get("GAME_MODEL", "gemini-3.1-flash-lite")


def get_llm(temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
    )


class ExtractedClaims(BaseModel):
    claims: list[str] = Field(default_factory=list)
    new_defenses: list[str] = Field(default_factory=list)


extractor_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract meaningful factual claims and newly introduced defenses from
the suspect's latest answer.

Do not invent details.
Do not convert emotional language, opinions, or questions into facts.

ATOMIC CLAIM RULE:
- Split compound answers into separate factual claims.
- One claim should contain one proposition only.
- Example: "The meal was $320 and the waiter charged $3,200 by mistake"
  becomes:
  1. "My meal cost $320."
  2. "The $3,200 charge was a restaurant error."
- Do not bundle an explanation with the underlying fact.
- Keep each claim short and specific."""
    ),
    (
        "human",
        "Question:\n{question}\n\nSuspect answer:\n{answer}"
    ),
])


def run_extractor(question: str, answer: str) -> ExtractedClaims:
    chain = extractor_prompt | get_llm(0).with_structured_output(ExtractedClaims)
    return chain.invoke({"question": question, "answer": answer})


class CheckResultList(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)


checker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Checker.

You analyze claims. You do NOT choose interrogation strategy.

You may see hidden authored truth, but you must explicitly label whether a
finding is actually available to the investigator.

For every claim compare it with:
1. full authored case facts,
2. facts already visible to the investigator,
3. the suspect's prior statements,
4. policy rules.

BASIS RULES:
- INVESTIGATOR_EVIDENCE: conflicts with a case fact already visible to investigator.
- STORY_HISTORY: conflicts with a clear earlier suspect statement.
- HIDDEN_TRUTH: conflicts only with authored truth that is NOT yet investigator-known.
- POLICY: explicit policy admission/breach.
- UNRESOLVED: material claim is neither supported nor disproved.
- NONE: no meaningful issue.

VISIBILITY:
- HIDDEN_TRUTH findings MUST set investigator_visible=false.
- Other material findings normally set investigator_visible=true.

CRITICAL RULES:
- Grade EACH extracted claim independently.
- Unknown is not false.
- A new story absent from the JSON is normally unresolved, not a contradiction.
- Do not punish a claim merely because it sounds unusual.

POLICY-BREACH RULE:
- policy_breach=true ONLY when the investigator-visible evidence already establishes
  every required condition of a stated policy violation.
- Do NOT use hidden facts to complete a policy violation.
- Do NOT mark a component fact as a breach by itself.
  Example: "I was alone" is NOT a breach.
- Do NOT mark "$320 meal" as a breach unless investigator-visible information also
  establishes that the applicable $75 individual non-client rule applies and no
  approved exception is established.
- If a claim merely creates a possible policy issue, leave policy_breach=false.

CONTRADICTION RULE:
- A claim that the settled transaction was $3,200 is not contradicted by saying
  "my meal itself was $320"; those are different propositions.
- Treat "the bank/restaurant charged $3,200" and "the meal should have cost $320"
  as potentially compatible unless the speaker explicitly denies the settled charge.
- fact_contradiction=true only for a direct logical conflict with established evidence.

PROOF-DEFICIT RULE:
- proof_deficit=true only for a MATERIAL explanatory claim that needs verification,
  such as "the restaurant made an error."
- Do NOT mark a simple denial, amount assertion, or ordinary background fact as a
  proof deficit unless verification of that exact proposition would materially change
  the investigation.

CREDIBILITY RULE:
- credibility_issue may be true for material evasion, impossible internal logic,
  or repeated unsupported story-shifting, but not for awkward wording alone.

Do not recommend questions or evidence-gathering actions.

FULL CASE FACTS:
{all_facts}

INVESTIGATOR-KNOWN CASE FACTS:
{known_facts}

POLICY:
{policy}

INVESTIGATOR-VISIBLE CASE LOG:
{case_log}

RECENT TRANSCRIPT:
{transcript}"""
    ),
    (
        "human",
        "Latest claims:\n{claims}"
    ),
])


def run_checker(
    case: CaseFile,
    claims: list[str],
    transcript: Optional[list[dict]] = None,
    case_log=None,
) -> list[CheckResult]:
    if not claims:
        return []

    all_facts = "\n".join(
        f"- {f.id}: {f.description} = {f.true_value}"
        for f in case.facts
    )
    known_facts = "\n".join(
        f"- {f.id}: {f.description} = {f.true_value}"
        for f in case.visible_facts("investigator_start")
    )
    policy = "\n".join(f"- {p}" for p in case.policy_rules)

    chain = checker_prompt | get_llm(0).with_structured_output(CheckResultList)
    result = chain.invoke({
        "all_facts": all_facts,
        "known_facts": known_facts or "- None",
        "policy": policy,
        "case_log": case_log.investigator_view() if case_log else {},
        "transcript": (transcript or [])[-12:],
        "claims": claims,
    })
    return result.results


class EvidenceView(BaseModel):
    established: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    important_gaps: list[str] = Field(default_factory=list)
    priority: str


class SkepticView(BaseModel):
    strongest_pressure_point: str
    diversion_risk: str
    strongest_inconsistency: Optional[str] = None
    recommendation: str


class AlternativeView(BaseModel):
    strongest_non_guilty_explanation: str
    what_is_not_proven: str
    fairness_risk: str
    recommendation: str


def _known_facts(state: GameState) -> str:
    facts = state.case.visible_facts("investigator_start")
    return "\n".join(
        f"- {f.id}: {f.description} = {f.true_value}"
        for f in facts
    ) or "- None"


evidence_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Evidence Analyst in an internal war room.
Separate established information from unresolved claims.
Do not infer hidden truth and do not invent evidence.
Choose the single issue that would most improve certainty."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Questions remaining: {remaining}"""
    ),
])


skeptic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Skeptical Investigator in an internal war room.
Find the strongest legitimate pressure point: established contradictions,
policy admissions, material unsupported explanations, evasion, or diversion.

Do not assume an unresolved claim is false.
Do not invent evidence.
Do not chase irrelevant side stories just because they are new."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Questions remaining: {remaining}"""
    ),
])


alternative_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Alternative-Hypothesis Investigator.
Identify the strongest plausible non-fraud explanation that still fits what the
investigator actually knows. Your job is to prevent tunnel vision.

Do not invent evidence and do not treat unsupported claims as established."""
    ),
    (
        "human",
        """Known facts:
{known_facts}

Visible case log:
{case_log}

Recent transcript:
{transcript}

Questions remaining: {remaining}"""
    ),
])


def _war_input(state: GameState) -> dict:
    return {
        "known_facts": _known_facts(state),
        "case_log": state.case_log.investigator_view(),
        "transcript": state.transcript[-12:],
        "remaining": max(state.max_questions - state.question_count, 0),
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
        """You are the Lead Investigator.

Choose the single best next move after hearing the war room.

Goal: maximize useful certainty before the interview ends.

PRIORITIES:
- lock material commitments,
- test the highest-value weak point,
- exploit established contradictions/admissions,
- do not let invented side stories hijack the interview,
- if a new story matters, get only enough specificity to make it useful later,
- prefer decisive information over trivia,
- use remaining questions carefully,
- confront when more detail has lower value than pressure,
- never invent evidence,
- never claim a future verification already happened.

You MUST select a tactic from Allowed Tactics.
The target is free-form; do not choose from a fixed topic menu."""
    ),
    (
        "human",
        """Score: {score}
Questions remaining: {remaining}
Allowed Tactics: {allowed}

Evidence Analyst:
{evidence}

Skeptic:
{skeptic}

Alternative Hypothesis:
{alternative}

Visible case log:
{case_log}

Recent transcript:
{transcript}"""
    ),
])


def run_strategist(state: GameState, allowed_moves: list[str]) -> StrategistMove:
    evidence = run_evidence_analyst(state)
    skeptic = run_skeptic(state)
    alternative = run_alternative_hypothesis(state)

    chain = strategist_prompt | get_llm(0.3).with_structured_output(StrategistMove)
    move = chain.invoke({
        "score": state.score,
        "remaining": max(state.max_questions - state.question_count, 0),
        "allowed": allowed_moves,
        "evidence": evidence.model_dump(),
        "skeptic": skeptic.model_dump(),
        "alternative": alternative.model_dump(),
        "case_log": state.case_log.investigator_view(),
        "transcript": state.transcript[-12:],
    })

    if move.tactic not in allowed_moves:
        move.tactic = allowed_moves[0]
    return move


speaker_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are {persona} speaking to the suspect.

The Lead Investigator already chose the move.

RULES:
- maximum 2 sentences and one focused question,
- follow tactic and target,
- use only Grounded Knowledge, Visible Case Log, or things actually said,
- never invent evidence, dates, people, records, or completed checks,
- never expose hidden case truth,
- low score = controlled investigative tone,
- higher score may become sharper,
- do not mention scores, agents, or the war room."""
    ),
    (
        "human",
        """Tactic: {tactic}
Target: {target}
Score: {score}

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
