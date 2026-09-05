"""
Post-interview Resolution Agent.

This agent may see full authored truth, but the hidden `guilty` flag is a
benchmark label, NOT evidence. The agent must never fabricate decisive evidence
just to make the benchmark label come true.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from models import GameState, ResolutionReport

load_dotenv()
MODEL_NAME = os.environ.get("GAME_MODEL", "gemini-3.1-flash-lite")


def get_llm(temperature: float = 0.2):
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)


resolution_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the post-interview Resolution Agent.

The interview is over. Resolve important pending claims as far as the authored
world actually allows, then produce the final game outcome.

STRICT RULES:
- The hidden guilty label is NOT evidence and MUST NOT justify the outcome.
- Do not invent CCTV, witnesses, emails, receipts, logs, or confessions that are
  not supported by the authored facts or interview record.
- If an open-world claim is neither supported nor contradicted by authored facts,
  its verification result is INCONCLUSIVE.
- If a claim directly conflicts with authored facts, it may be DISPROVED.
- If it matches authored facts, it may be CONFIRMED.
- A vague or unsupported story does not become exculpatory merely because it is
  inconclusive.
- Likewise, inconclusive claims are not proof of guilt.
- Judge the total evidentiary picture: established contradictions, admissions,
  policy breaches, credibility problems, and what remains genuinely uncertain.

OUTCOMES:
CAUGHT:
  evidence is strong enough to justify intentional wrongdoing.
NOT_PROVEN:
  suspicion may remain, but intentional wrongdoing is not established strongly enough.
POLICY_VIOLATION_ONLY:
  non-compliance is established, but intentional fraud is not.

AFTERMATH:
Write a short realistic after-investigation story. You may say investigators
reviewed/checked relevant records in general terms, but any decisive result must
come from the authored facts. Do not invent specific evidence sources."""
    ),
    (
        "human",
        """FULL AUTHORED FACTS:
{facts}

POLICY:
{policy}

FULL INTERNAL CASE LOG:
{case_log}

TRANSCRIPT:
{transcript}

INTERVIEW SCORE:
{score}

HIDDEN BENCHMARK LABEL (NOT EVIDENCE):
{guilty_label}"""
    ),
])


def run_resolution(state: GameState) -> ResolutionReport:
    facts = "\n".join(
        f"- {f.id}: {f.description} = {f.true_value}"
        for f in state.case.facts
    )
    policy = "\n".join(f"- {p}" for p in state.case.policy_rules)

    chain = resolution_prompt | get_llm(0.2).with_structured_output(ResolutionReport)
    return chain.invoke({
        "facts": facts,
        "policy": policy,
        "case_log": state.case_log.model_dump(),
        "transcript": state.transcript,
        "score": state.score,
        "guilty_label": state.case.guilty,
    })


def format_resolution(report: ResolutionReport) -> str:
    return (
        f"FINAL OUTCOME: {report.outcome.value}\n\n"
        f"{report.aftermath}\n\n"
        f"Reasoning: {report.reasoning}\n"
        f"Confidence: {report.confidence}"
    )
