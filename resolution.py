"""
Post-interview Resolution Agent.

The agent may see full authored truth, but the hidden guilty flag is a benchmark
label, not evidence. Post-interview scoring only adds newly verified incriminating
value from claims that could not be established during the interview.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models import (
    ClaimStatus,
    EvidentiaryImpact,
    FinalOutcome,
    GameState,
    ResolutionReport,
    VerificationResult,
    VerificationStatus,
    format_facts,
)


class ResolutionDraft(BaseModel):
    """LLM-facing shape only. verification_score_delta and final_score are
    always computed in code from `verifications` (see _verification_delta),
    so they are never part of the schema the model has to fill in — asking
    for them here would just be paying output tokens to guess a number that
    gets thrown away."""

    verifications: list[VerificationResult] = Field(default_factory=list)
    outcome: FinalOutcome
    confidence: str
    reasoning: str
    aftermath: str

load_dotenv()
MODEL_NAME = os.environ.get("GAME_MODEL", "gemini-3.1-flash-lite")


IMPACT_POINTS = {
    EvidentiaryImpact.NONE: 0,
    EvidentiaryImpact.WEAK: 4,
    EvidentiaryImpact.MODERATE: 8,
    EvidentiaryImpact.STRONG: 15,
    EvidentiaryImpact.DECISIVE: 25,
}


def get_llm(temperature: float = 0.2):
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)


resolution_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the post-interview Resolution Agent.

Resolve important UNVERIFIED/PENDING claims only as far as the authored world allows.
The hidden guilty label is NOT evidence.

ALREADY-ESTABLISHED FINDINGS BELOW ALREADY CONTRIBUTED TO THE INTERVIEW SCORE.
Do not re-verify them or assign them evidentiary_impact again — that would double-count
the same fact. If your reasoning needs to reference one, use status=NOT_MATERIAL and
evidentiary_impact=none for it. Only assign nonzero evidentiary_impact to a claim that
was genuinely left UNVERIFIED or PENDING by the interview.

STRICT RULES:
- Do not invent CCTV, witnesses, emails, receipts, logs, merchant responses, or
  confessions not supported by authored facts or interview record.
- If an open-world claim is neither supported nor contradicted by authored facts,
  mark it INCONCLUSIVE.
- If authored facts directly conflict with it, it may be DISPROVED.
- If authored facts directly support it, it may be CONFIRMED.
- Inconclusive is neither guilt nor innocence.

EVIDENTIARY IMPACT:
For each verification, evidentiary_impact means NEW incriminating value toward
intentional wrongdoing created by the verification.
- Confirming an innocent/exculpatory defense => none.
- Inconclusive => none.
- Disproving a central defense may be moderate/strong/decisive depending on how
  directly it bears on intentional wrongdoing.
- Do not inflate impact just to reach the threshold.
- Each authored fact below is tagged [weight=..., certainty=...]. A
  disproof resting only on a certainty=slow fact (a review finding, an
  inference, a secondhand statement rather than a primary record) cannot be
  scored above moderate, no matter how central the claim looks.

Suggest an outcome based on the evidentiary picture, but the game engine will enforce
that CAUGHT requires the final numeric case score to reach the authored threshold.

AFTERMATH:
Write a short realistic after-investigation story. General checking is fine, but any
specific decisive result must come from authored facts."""
    ),
    (
        "human",
        """FULL AUTHORED FACTS:
{facts}

POLICY:
{policy}

ALREADY-ESTABLISHED FINDINGS (already scored — do not re-verify):
{already_established}

FULL INTERNAL CASE LOG:
{case_log}

TRANSCRIPT:
{transcript}

INTERVIEW SCORE:
{score}/{threshold}

HIDDEN BENCHMARK LABEL (NOT EVIDENCE):
{guilty_label}"""
    ),
])


def _verification_delta(report: ResolutionReport) -> int:
    """Only newly disproved pending claims add post-interview incriminating points."""
    seen: set[str] = set()
    total = 0
    for result in report.verifications:
        key = " ".join(result.claim.lower().split())
        if key in seen:
            continue
        seen.add(key)
        if result.status == VerificationStatus.DISPROVED:
            total += IMPACT_POINTS[result.evidentiary_impact]
    return total


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def run_resolution(state: GameState) -> ResolutionReport:
    facts = format_facts(state.case.facts)
    policy = "\n".join(f"- {p}" for p in state.case.policy_rules)

    # Anything the interview already resolved to a non-unverified status
    # already contributed its evidentiary_impact to state.score. Without this,
    # the Resolution Agent has no way to know a claim was already scored and
    # can independently "confirm" the same underlying fact again, double-
    # counting it (observed in practice: a contradiction already worth +30
    # in-interview got re-verified and added again post-interview).
    already_established = [
        c.text for c in state.case_log.claims
        if c.status != ClaimStatus.UNVERIFIED and c.investigator_visible
    ]

    chain = resolution_prompt | get_llm(0.2).with_structured_output(ResolutionDraft)
    draft = chain.invoke({
        "facts": facts,
        "policy": policy,
        "already_established": "\n".join(f"- {t}" for t in already_established) or "- None",
        "case_log": state.case_log.model_dump(),
        "transcript": state.transcript,
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "guilty_label": state.case.guilty,
    })

    # The prompt already tells the model "confirming/inconclusive => none
    # impact", but a rule stated only in the prompt can silently drift on any
    # given call. Force it here so a CONFIRMED or INCONCLUSIVE verification
    # can never carry a nonzero evidentiary_impact regardless of what the LLM
    # produced — only a DISPROVED verification is allowed to matter.
    for v in draft.verifications:
        if v.status != VerificationStatus.DISPROVED:
            v.evidentiary_impact = EvidentiaryImpact.NONE

    # Code-side backstop for the same rule the prompt states above: even if
    # the model restates an already-established finding as a fresh
    # verification, it cannot carry additional points. Exact-match or
    # one-contains-the-other on normalized text, same tolerance the rest of
    # this codebase uses for claim dedup (see game_logic._claim_key).
    established_norm = [_normalize(t) for t in already_established]
    for v in draft.verifications:
        v_norm = _normalize(v.claim)
        if any(
            v_norm == est or (len(est) > 20 and (est in v_norm or v_norm in est))
            for est in established_norm
        ):
            v.evidentiary_impact = EvidentiaryImpact.NONE

    report = ResolutionReport(
        verifications=draft.verifications,
        outcome=draft.outcome,
        confidence=draft.confidence,
        reasoning=draft.reasoning,
        aftermath=draft.aftermath,
    )

    delta = _verification_delta(report)
    report.verification_score_delta = delta
    report.final_score = state.score + delta

    # Keep the numeric KPI and ending parallel: CAUGHT requires the threshold.
    if report.final_score >= state.case.arrest_threshold:
        report.outcome = FinalOutcome.CAUGHT
    elif report.outcome == FinalOutcome.CAUGHT:
        report.outcome = FinalOutcome.NOT_PROVEN

    return report


def format_resolution(report: ResolutionReport) -> str:
    return (
        f"FINAL OUTCOME: {report.outcome.value}\n\n"
        f"{report.aftermath}\n\n"
        f"Reasoning: {report.reasoning}\n"
        f"Verification score change: +{report.verification_score_delta}\n"
        f"Final case score: {report.final_score}\n"
        f"Confidence: {report.confidence}"
    )
