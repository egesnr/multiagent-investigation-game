"""
Post-interview Resolution Agent.

The agent may see full authored truth, but the hidden guilty flag is a benchmark
label, not evidence. Post-interview scoring only adds newly verified incriminating
value from claims that could not be established during the interview.

Two LLM calls, not one, and in this order on purpose: verifications are scored
first, the numeric outcome is finalized in code, and only THEN is the narrative
written — grounded in the outcome that's actually going to be shown. A single
combined call was writing an aftermath like "terminated for gross misconduct"
in the same breath as an outcome that the code then downgraded to NOT_PROVEN
because the score fell short of the threshold, since the model had no way to
know its own draft outcome would be overridden.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_utils import invoke_with_retry
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


class VerificationDraft(BaseModel):
    verifications: list[VerificationResult] = Field(default_factory=list)
    outcome_recommendation: FinalOutcome = Field(
        description="Your read of the evidentiary picture alone. The game "
        "engine will independently enforce that CAUGHT requires the final "
        "numeric score to reach the authored threshold — if it doesn't, this "
        "recommendation is downgraded to NOT_PROVEN regardless of what you "
        "pick here, so pick honestly rather than trying to game the check."
    )


class NarrativeDraft(BaseModel):
    confidence: str = Field(
        description="How confident the finding is, as a word: high, moderate, "
        "or low. Not a score or a number."
    )
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


verification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the post-interview Resolution Agent, verification stage.

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
  scored above moderate, no matter how central the claim looks."""
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


narrative_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the post-interview Resolution Agent, narrative stage.

The verification pass is already complete and the FINAL OUTCOME below is
already decided by the game engine — it is not yours to choose or hedge
against. Write reasoning and an aftermath that are consistent with that exact
outcome:
- CAUGHT: the case cleared the threshold. Write it as a case that stood up.
- NOT_PROVEN: the case did not clear the threshold, whatever suspicion
  remains. Do not write an aftermath implying termination, confession, or a
  closed fraud case — write a realistic account of an investigation that
  raised concerns without reaching a provable conclusion.
- POLICY_VIOLATION_ONLY: real policy breaches were established, but not
  intentional fraud. Write consequences proportionate to a policy violation
  (a warning, discipline, repayment), not a fraud termination.

Any specific decisive detail must come from the authored facts or verifications
below, not invented. General/vague language ("standard review procedures
continue") is fine when the record doesn't support anything more specific."""
    ),
    (
        "human",
        """FINAL OUTCOME (fixed, do not contradict): {outcome}
Final case score: {final_score}/{threshold}

AUTHORED FACTS (the only source for any specific figure, date, name or
document you mention — quote them exactly, never reconstruct from memory):
{facts}

VERIFICATIONS FROM THIS STAGE:
{verifications}

ALREADY-ESTABLISHED FINDINGS FROM THE INTERVIEW:
{already_established}

FULL INTERNAL CASE LOG:
{case_log}"""
    ),
])


def _verification_delta(verifications: list[VerificationResult]) -> int:
    """Only newly disproved pending claims add post-interview incriminating points."""
    seen: set[str] = set()
    total = 0
    for result in verifications:
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
    already_established_text = "\n".join(f"- {t}" for t in already_established) or "- None"

    verification_chain = (
        verification_prompt | get_llm(0.2).with_structured_output(VerificationDraft)
    )
    verification_draft = invoke_with_retry(verification_chain, {
        "facts": facts,
        "policy": policy,
        "already_established": already_established_text,
        "case_log": state.case_log.model_dump(),
        "transcript": state.transcript,
        "score": state.score,
        "threshold": state.case.arrest_threshold,
        "guilty_label": state.case.guilty,
    })

    verifications = verification_draft.verifications

    # The prompt already tells the model "confirming/inconclusive => none
    # impact", but a rule stated only in the prompt can silently drift on any
    # given call. Force it here so a CONFIRMED or INCONCLUSIVE verification
    # can never carry a nonzero evidentiary_impact regardless of what the LLM
    # produced — only a DISPROVED verification is allowed to matter.
    for v in verifications:
        if v.status != VerificationStatus.DISPROVED:
            v.evidentiary_impact = EvidentiaryImpact.NONE

    # Code-side backstop for the same rule the prompt states above: even if
    # the model restates an already-established finding as a fresh
    # verification, it cannot carry additional points. Exact-match or
    # one-contains-the-other on normalized text, same tolerance the rest of
    # this codebase uses for claim dedup (see game_logic._claim_key).
    established_norm = [_normalize(t) for t in already_established]
    for v in verifications:
        v_norm = _normalize(v.claim)
        if any(
            v_norm == est or (len(est) > 20 and (est in v_norm or v_norm in est))
            for est in established_norm
        ):
            v.evidentiary_impact = EvidentiaryImpact.NONE

    delta = _verification_delta(verifications)

    # Give back suspicion points for excuses that turned out to be true. In the
    # room an unsupported defense raises suspicion (game_logic.PROVISIONAL_POINTS)
    # because a liar must not profit from saying things nobody can check on the
    # spot — but a suspect who was telling the truth all along should not be
    # left carrying that penalty once verification confirms them.
    refund = 0
    for v in verifications:
        if v.status != VerificationStatus.CONFIRMED:
            continue
        v_norm = _normalize(v.claim)
        for claim_text, points in state.provisional_findings.items():
            c_norm = _normalize(claim_text)
            if v_norm == c_norm or (len(c_norm) > 20 and (c_norm in v_norm or v_norm in c_norm)):
                refund += points
                break

    final_score = state.score + delta - refund

    # Keep the numeric KPI and ending parallel: CAUGHT requires the threshold.
    # This is decided BEFORE the narrative is written, not after, so the
    # aftermath text is never generated against an outcome the code is about
    # to override.
    if final_score >= state.case.arrest_threshold:
        outcome = FinalOutcome.CAUGHT
    elif verification_draft.outcome_recommendation == FinalOutcome.CAUGHT:
        outcome = FinalOutcome.NOT_PROVEN
    else:
        outcome = verification_draft.outcome_recommendation

    narrative_chain = narrative_prompt | get_llm(0.3).with_structured_output(NarrativeDraft)
    narrative = invoke_with_retry(narrative_chain, {
        "outcome": outcome.value,
        "final_score": final_score,
        "threshold": state.case.arrest_threshold,
        # The narrative stage was told to ground every specific detail in the
        # authored facts "below" while never actually being passed them, so it
        # reconstructed figures from surrounding context and got them wrong
        # (reported the charge as $3,150 against an authored $3,200).
        "facts": facts,
        "verifications": [v.model_dump() for v in verifications] or "- None",
        "already_established": already_established_text,
        "case_log": state.case_log.model_dump(),
    })

    return ResolutionReport(
        verifications=verifications,
        outcome=outcome,
        confidence=narrative.confidence,
        reasoning=narrative.reasoning,
        aftermath=narrative.aftermath,
        verification_score_delta=delta,
        final_score=final_score,
    )


def format_resolution(report: ResolutionReport) -> str:
    return (
        f"FINAL OUTCOME: {report.outcome.value}\n\n"
        f"{report.aftermath}\n\n"
        f"Reasoning: {report.reasoning}\n"
        f"Verification score change: +{report.verification_score_delta}\n"
        f"Final case score: {report.final_score}\n"
        f"Confidence: {report.confidence}"
    )
