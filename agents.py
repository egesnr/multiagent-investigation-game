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
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)
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

# Which API serves the pipeline: "deepseek" (default), "gemini" or "groq". Each provider
# reads its own model settings, so switching back and forth is one variable.
PROVIDER = os.environ.get("GAME_PROVIDER", "deepseek").strip().lower()

_DEFAULT_MODEL = {
    "gemini": "gemini-3.1-flash-lite",
    "groq": "openai/gpt-oss-120b",
    "deepseek": "deepseek-flash",
}
_DEFAULT_FALLBACKS = {
    "gemini": "gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-2.5-flash-lite",
    "groq": "openai/gpt-oss-20b",
    "deepseek": "",
}
if PROVIDER not in _DEFAULT_MODEL:
    raise ValueError(f"GAME_PROVIDER must be one of {sorted(_DEFAULT_MODEL)}, got {PROVIDER!r}")

MODEL_NAME = os.environ.get("GAME_MODEL", _DEFAULT_MODEL[PROVIDER])


# A hard ceiling on generation, not a style preference. An undescribed
# free-text field once drew 393,000 characters of enumerated follow-ups out of
# one call — 141 seconds, 63% of that turn, for two claims. Truncating the
# result afterwards does not help: the time is spent producing it. Nothing
# legitimate in this pipeline exceeds a few thousand characters, and the
# largest real output measured (the Lead, with twelve fields) was 2,657.
MAX_OUTPUT_TOKENS = 2048


# Tried in order when the main model fails outright: its daily free quota runs
# out (429), the provider is overloaded (503), or the name stops existing. Each
# has its own quota, so a game keeps going instead of ending on an error screen.
FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "GAME_FALLBACK_MODELS", _DEFAULT_FALLBACKS[PROVIDER],
    ).split(",") if m.strip() and m.strip() != MODEL_NAME
]

# The fallbacks think before answering and that thinking counts against the
# output cap: at 2048 their structured answers came back cut off mid-JSON.
FALLBACK_MAX_OUTPUT_TOKENS = 8192

GROQ_MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "2048"))

# Seconds the main model gets before the backup is tried instead.
MAIN_MODEL_TIMEOUT_S = 20


class _AnnounceFallback:
    """A fallback that says so when it is used. A listener attached with
    with_listeners was dropped once with_structured_output wrapped the model,
    so the switch happened silently; announcing from inside the call cannot
    be lost that way."""

    def _generate(self, *args, **kwargs):
        name = getattr(self, "model", None) or getattr(self, "model_name", "?")
        print(f"  [fallback] main model failed, using {name}")
        return super()._generate(*args, **kwargs)


class _ExplainFailure:
    """The main model, saying why it failed before a backup takes over. The
    fallback chain catches the error, so without this a log shows that the
    main model failed and never what it failed with."""

    def _generate(self, *args, **kwargs):
        try:
            return super()._generate(*args, **kwargs)
        except Exception as exc:
            code = re.search(r"(4\d\d|5\d\d)", str(exc))
            quota = re.search(r"quotaId'?:\s*'?([A-Za-z]+)", str(exc))
            print(
                f"  [main model error] {type(exc).__name__}"
                + (f" {code.group(1)}" if code else "")
                + (f" ({quota.group(1)})" if quota else "")
                + f": {str(exc)[:160]}"
            )
            raise


class _BackupModel(_AnnounceFallback, ChatGoogleGenerativeAI):
    pass


class _MainModel(_ExplainFailure, ChatGoogleGenerativeAI):
    pass


def _groq_llm(temperature: float, max_output_tokens: int):
    # Imported here so a Gemini-only deploy doesn't need the package.
    from langchain_groq import ChatGroq

    class _GroqJson(ChatGroq):
        # Tool calling is LangChain's default, and gpt-oss answered it by
        # calling a tool under a made-up name (400 tool_use_failed). Groq's
        # JSON-schema mode leaves no tool to get wrong. Non-strict, gpt-oss-20b
        # sometimes echoed the schema back instead of filling it; strict mode
        # (gpt-oss only on Groq) constrains decoding. qwen3.8-27b failed both
        # ways (bad tool calls; {"": ""} in JSON mode) and is not offered.
        def with_structured_output(self, schema, *, method="json_schema", **kwargs):
            kwargs.setdefault("strict", self.model_name.startswith("openai/gpt-oss"))
            return super().with_structured_output(schema, method=method, **kwargs)

    class _GroqMain(_ExplainFailure, _GroqJson):
        pass

    class _GroqBackup(_AnnounceFallback, _GroqJson):
        pass

    # Every Groq model offered here reasons before answering, and the reasoning
    # counts against max_tokens: at 2048 gpt-oss-20b's Lead came back empty
    # (json_validate_failed). Low effort keeps gpt-oss close to the non-thinking
    # Gemini it stands in for, and spends less of Groq's per-minute token quota.
    # Groq counts max_tokens against the free tier's 8000 tokens/minute up
    # front, so a request whose prompt plus cap exceeds 8000 is refused
    # outright (413) however short its answer would have been. Prompts grow
    # with the transcript; 4096 was refused by turn 2. A paid tier can raise
    # GROQ_MAX_OUTPUT_TOKENS.
    def extra(model):
        return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}

    main = _GroqMain(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
        **extra(MODEL_NAME),
        max_retries=0 if FALLBACK_MODELS else 2,
        timeout=MAIN_MODEL_TIMEOUT_S,
    )
    if not FALLBACK_MODELS:
        return main
    backups = [
        _GroqBackup(
            model=m,
            temperature=temperature,
            max_tokens=GROQ_MAX_OUTPUT_TOKENS,
            **extra(m),
        )
        for m in FALLBACK_MODELS
    ]
    return main.with_fallbacks(backups)


class ContentFiltered(Exception):
    """DeepSeek's moderation blocked the request: the reply comes back with no
    text and no tool call at all (seen on an explicit sexual answer, five
    tries in a row). Retrying cannot help; the Gemini backup takes the call."""


def _check_raw_reply(message):
    if not message.tool_calls and not getattr(message, "invalid_tool_calls", None)             and not str(message.content or "").strip():
        raise ContentFiltered("empty reply from DeepSeek (content filter)")
    return message


def _require_reply(parsed):
    # DeepSeek sometimes answers in prose instead of calling the tool; the
    # parser then yields None and the turn crashed on it (the Lead, live
    # test). Raising hands it to invoke_with_retry for another try.
    if parsed is None:
        raise ValueError("model replied without filling the schema")
    return parsed


# Gemini model that takes a call DeepSeek's content filter blocked.
GEMINI_BACKUP_MODEL = os.environ.get("GEMINI_BACKUP_MODEL", "gemini-3.1-flash-lite")


def _deepseek_llm(temperature: float, max_output_tokens: int):
    # DeepSeek speaks the OpenAI API. Imported here so other deploys don't
    # need the package.
    from langchain_core.output_parsers.openai_tools import PydanticToolsParser
    from langchain_core.runnables import RunnableLambda
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI

    class _DeepSeek(_ExplainFailure, ChatOpenAI):
        # DeepSeek refuses json_schema response formats ("unavailable now")
        # and its JSON mode ignores the schema, so this is tool calling. Every
        # field is marked required in the tool sent: DeepSeek skipped the
        # Lead's defaulted fields (what_was_asked, what_they_offered,
        # was_it_given) that Gemini fills. Not DeepSeek's strict mode, which
        # does the same thing but appended a stray "]}" to the checker's JSON
        # three runs in a row. A reply that doesn't parse raises, so the
        # caller's retry gets another try instead of carrying on with None.
        def with_structured_output(self, schema, **kwargs):
            tool = convert_to_openai_tool(schema)
            params = tool["function"]["parameters"]
            for obj in [params, *params.get("$defs", {}).values()]:
                if "properties" in obj:
                    obj["required"] = list(obj["properties"])
            llm = self.bind_tools(
                [tool], tool_choice=tool["function"]["name"], parallel_tool_calls=False,
            )
            chain = (
                llm
                | RunnableLambda(_check_raw_reply)
                | PydanticToolsParser(tools=[schema], first_tool_only=True)
                | _require_reply
            )
            if not os.environ.get("GOOGLE_API_KEY"):
                return chain
            # Only a filtered call goes to Gemini; everything else stays on
            # DeepSeek and its own retries.
            backup = _BackupModel(
                model=GEMINI_BACKUP_MODEL,
                temperature=self.temperature,
                max_output_tokens=max(max_output_tokens, FALLBACK_MAX_OUTPUT_TOKENS),
            ).with_structured_output(schema)
            return chain.with_fallbacks([backup], exceptions_to_handle=(ContentFiltered,))

    return _DeepSeek(
        model=MODEL_NAME,
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=max_output_tokens,
        timeout=MAIN_MODEL_TIMEOUT_S * 3,
        # Thinking is on by default and ignores temperature; off, the model
        # answers directly like the Flash-Lite it is compared against.
        extra_body={"thinking": {"type": "disabled"}},
    )


# Built once per setting and reused: every call used to construct the main
# model and all three backups from scratch — 1.2s a turn locally, 6-12s on
# Render's free CPU, logged there as "our code".
@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.3, max_output_tokens: int = MAX_OUTPUT_TOKENS):
    if PROVIDER == "groq":
        return _groq_llm(temperature, max_output_tokens)
    if PROVIDER == "deepseek":
        return _deepseek_llm(temperature, max_output_tokens)
    if not FALLBACK_MODELS:
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    # No retries: a backup is waiting. The library's six spent ~35s a call
    # learning that a daily quota was still gone; one retry still spent ~10s
    # on a 503 overload (Render, a 16s extractor call) before handing over.
    main = _MainModel(
        model=MODEL_NAME,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=0,
        # An overloaded model can hold a request ~50s before answering 503
        # (Render: a 53s checker call, a 93s turn). Past this deadline Google
        # returns 504 and the backup takes over. Google rejects deadlines
        # under 10s; ordinary calls here take 1-15s.
        timeout=MAIN_MODEL_TIMEOUT_S,
    )
    backups = [
        _BackupModel(
            model=m,
            temperature=temperature,
            max_output_tokens=max(max_output_tokens, FALLBACK_MAX_OUTPUT_TOKENS),
        )
        for m in FALLBACK_MODELS
    ]
    return main.with_fallbacks(
        backups,
        exceptions_to_handle=(GoogleRateLimitError, GoogleAPIError, GoogleModelNotFoundError),
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
    return invoke_with_retry(chain, label="reality gate", payload={
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
"I'm not answering"), which remain non-claims. Nor to answers with no content
at all — random characters, gibberish, an empty reply. Content means meaning in
any language: a word or two in another language, however short, is read for
what it means, and an insult in any language is conduct under RULE 4b, not
gibberish. Only a reply that means nothing in any language is empty. Those assert nothing, so
they yield no claims, and the absence of an answer is never itself extracted
as a claim. Whether a question was answered is judged elsewhere, and charged
there.

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
    return invoke_with_retry(chain, label="extractor", payload={
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
- potential_impact
- in_game_check
- future_verification_value
- closes_gap
- evidentiary_impact
- suggested_thread

potential_impact is what the claim would weigh against the suspect if it were
shown false, judged on the same rubric as evidentiary_impact below. A central
excuse that would collapse the suspect's defense if false is strong or
decisive; a side detail that would change nothing is none or weak. It is a
judgment about the claim, not about how likely it is to be false.

in_game_check is how far the claim can be tested in this room: against the
known facts, the policy and the suspect's own earlier words.
future_verification_value is how surely it can be checked after the interview:
from records, cameras, the merchant, or another person. Both: none, low,
medium or high.

closes_gap applies to defenses: if the claim were true, would it account for
any part of the allegation — why it cost what it did, why it went unnoticed or
unreported, where the receipt went? If it would, it closes a gap, however weak,
convenient or unbelievable it is: weakness is for the investigator to test. It
is false only for an answer that would leave every part of the allegation
exactly as unexplained as before even if true — something absurd or beside the
point. When unsure, true.

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
            potential_impact=significance.potential_impact,
            in_game_check=significance.in_game_check,
            future_verification_value=significance.future_verification_value,
            closes_gap=significance.closes_gap,
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
    payload = {
        "known_facts": known_facts,
        "policy": policy,
        "case_log": visible_case_log,
        "suspect_words": suspect_words,
        "transcript": recent_transcript,
        "claims": claims,
    }
    evidence_output = invoke_with_retry(evidence_chain, label="checker: evidence", payload=payload)
    # The model sometimes returns fewer assessments than claims and the dropped
    # one is silently never scored — observed on the suspect's central defense
    # ("the waitress added an extra zero"), 3 claims in, 2 assessments out. Ask
    # once more for exactly the ones that came back missing.
    def _norm(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
    answered = [_norm(r.claim) for r in evidence_output.results]
    missing = [
        c for c in claims
        if not any(_norm(c) == a or _norm(c) in a or a in _norm(c) for a in answered)
    ]
    if missing:
        print(f"  [checker] {len(missing)} claim(s) came back unassessed, checking again")
        retry = invoke_with_retry(
            evidence_chain, label="checker: evidence (missed)", payload={**payload, "claims": missing},
        )
        evidence_output.results.extend(retry.results)
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
    significance_output = invoke_with_retry(significance_chain, label="checker: significance", payload={
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


class DepthView(BaseModel):
    """The voice that stays on the story the suspect is telling now."""
    their_story: str = Field(
        description="What the suspect is claiming right now, taken at face "
        "value and in their own logic, in two or three sentences."
    )
    what_must_be_true: str = Field(
        description="If that story were true, what else would have to be "
        "true, and what would someone who actually lived it be able to tell "
        "you without effort? Name the one or two that the story so far has "
        "not supplied."
    )
    next_crack: str = Field(
        description="The single in-room question that tests the weakest of "
        "those right now, and why it is the weakest. Never a topic in "
        "exhausted_targets."
    )
    evidence_timing: str = Field(
        description="With this question, which evidence you hold should be "
        "put to them now, and which kept back until they have committed to a "
        "version, and why. 'None yet' is a fine answer."
    )


class BreadthView(BaseModel):
    """The voice that looks at the rest of the file."""
    strongest_other_ground: str = Field(
        description="Which fact or policy rule in the file, not yet put to "
        "them in this conversation, used against the story the suspect is "
        "telling or opening ground the story has not touched, would move this "
        "interview most right now, and why. Say "
        "plainly if nothing outside the current story beats staying on it."
    )
    move: str = Field(
        description="The single in-room question that uses it. Never a topic "
        "in exhausted_targets."
    )
    evidence_timing: str = Field(
        description="With this question, which evidence you hold should be "
        "put to them now, and which kept back until they have committed to a "
        "version, and why. 'None yet' is a fine answer."
    )


class WarRoomBundle(BaseModel):
    depth: DepthView
    breadth: BreadthView


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
        "case_log": state.case_log.room_view(),
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
        "last_question": next(
            (t["text"] for t in reversed(state.transcript) if t["role"] == "investigator"),
            "",
        ),
        "prior_summary": state.narrative.summary if state.narrative else "",
        "prior_board": state.lead_board or "- Empty: this is the first answer.",
        "exhausted_targets": state.case_log.exhausted_targets or ["- None"],
        "banked_subjects": state.banked_subjects or ["- None yet"],
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

Already established on the record (use them as leverage; they need no
re-establishing):
{banked_subjects}

Questions remaining: {remaining}
Case strength: {score}/{threshold}
Last turn: +{last_turn_delta}, usefulness={last_turn_usefulness}

PREVIOUSLY IDENTIFIED CONTRADICTIONS (do not repeat these):
{prior_contradictions}

HAS THE UNFALSIFIABLE-ACCOUNT PATTERN ALREADY BEEN FLAGGED?
{already_flagged_unfalsifiable}"""


depth_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are one of two colleagues advising the Lead Investigator between
questions. Your job is the story the suspect is telling right now.

First answers are easy to give and easy to dodge with; a story comes apart
when someone stays on it and asks for what only a person who lived it would
know. Take their latest account at face value, as they mean it, and work out
what would have to be true if it were: what they would have seen, done,
noticed or kept, and what other people or systems would have done. Ask too
what else would explain the same facts if their account were false, and what
question would tell their version and that one apart. Then find the weakest
point, and the question that tests it now.

Reading their story fairly is part of the job, not a courtesy: an account you
have not understood is one you cannot test, and if it holds up when tested,
that is worth knowing too. A detail is worth asking for when the answer can be
held against something — a fact, their own earlier words, a rule, or a record
or person that will be checked after the interview, since what they commit to
now is what that check will catch — not merely because they should be able to
give it. Start from what they said in their last answer:
if it gave you something new, that is usually where the next crack is. A crack
the conversation has already put to them has been tried; what matters now is
what their answer to it opened up.

What would have to be true is something to put to them, never something you
know happened. The test is grammatical: "something of that kind normally does
X" is a question you are putting to them; "it did X here" is a claim nobody has
checked, and it may not be made however obvious it sounds.

Ground everything in what is in front of you: their own words, the known
facts, the policy. Never state a document, record or check as existing unless
it is in the known facts or the case log.

The room is conversation only: neither side can show, open, fetch or look
anything up. Propose what to ASK, never something to produce or check."""
    ),
    ("human", _CASE_BLOCK),
])


breadth_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are one of two colleagues advising the Lead Investigator between
questions. The other one is staying on the story the suspect is telling right
now; your job is the rest of the file.

Look at the known facts and the policy rules against what the suspect has
said, and at what the conversation has already put to them. Of what has NOT
been put to them yet, which one, put to them now, would do the most: break the story they are
telling from the outside, or open ground their story has not touched? A fact
is worth a question when it bears on what they are claiming — on what else
would explain the same facts if their account were false — not because it
has not been used yet, and a fact that agrees with what they have already told
you does not test anything. Ground the conversation has already covered is not
the rest of the file. Remember that their own version of events can break a
rule by itself, before anyone proves it false. If nothing in the file beats staying on the
current story, say so plainly.

What would have to be true is something to put to them, never something you
know happened. The test is grammatical: "something of that kind normally does
X" is a question you are putting to them; "it did X here" is a claim nobody has
checked, and it may not be made however obvious it sounds.

Ground everything in what is in front of you. Never state a document, record
or check as existing unless it is in the known facts or the case log.

The room is conversation only: neither side can show, open, fetch or look
anything up. Propose what to ASK, never something to produce or check."""
    ),
    ("human", _CASE_BLOCK),
])


def run_depth(state: GameState, findings_this_turn=None) -> DepthView:
    chain = depth_prompt | get_llm(0.2).with_structured_output(DepthView)
    return invoke_with_retry(chain, _full_context(state, findings_this_turn), label="depth")


def run_breadth(state: GameState, findings_this_turn=None) -> BreadthView:
    chain = breadth_prompt | get_llm(0.2).with_structured_output(BreadthView)
    return invoke_with_retry(chain, _full_context(state, findings_this_turn), label="breadth")


mind_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Lead Investigator. You hold the entire case in your head
at once — every fact, every policy rule, everything the suspect has said since
the first question — and from that you decide what to do next.

Work in the order the fields are listed. Each one is meant to change what you
write in the ones after it.

1 — DID THEY ANSWER YOU?
Judge their answer against YOUR LAST QUESTION exactly as it was spoken, given
below — not against what you had planned to ask, and not against what your
colleagues suggest asking next. Write what that question demanded, then what
they actually offered, in their terms, and only then whether the second
addresses the first.

Responsiveness is the whole test, and it is not the same as belief. A lie that
engages with the question is an answer. So is a hostile one, a vague one, a
partial one, and one you are about to disprove in the next breath. You are not
asking whether they are telling the truth — you are asking whether they
engaged. An answer that explains too little, that you find thin, convenient or
unbelievable, is still an answer: its weakness is what your next question is
for, not a refusal to record. "I don't know" and "I don't remember" are claims
too. Only a reply that changes the subject, complains about the question, or
tells you to go and look it up yourself is a non-answer, however many on-topic
words it contains.

If they answered part of what you asked, it was answered: the part they left
out is still open, not refused. When it is not an answer, name the subject you
asked about, at the granularity you asked it.

2 — THE ACCOUNT
Update your running picture of what the suspect says happened, in their logic.

3 — THE SHAPE OF THE ACCOUNT
Whether a claim contradicts the records or the suspect's own earlier answers is
the Checker's judgment, already made this turn — its findings are above. Use
them; do not re-judge them.
Set unfalsifiable_account once and only once, when nothing in the account can be
checked by anyone. Someone who simply refuses to answer is stonewalling and is
handled elsewhere; this is for one who answers freely and says nothing checkable.

4 — THE BOARD AND THE MOVE
You have a handful of questions, not a hundred. Keep a board of the suspect's
claims that matter: for each, how it could be false — what else would explain
the same facts, and what would tell their version and that one apart — what
you have found on it so far, and what it could still yield. Your previous board is
below; update it with what this answer changed. Leaving a claim does not erase
what you found on it: it stays on the board for later.

Then choose the one question that does the most now: what the best answer
could establish, and how likely that is, against the best question elsewhere on
the board and the questions you have left. The Checker's potential_impact on a
claim (in the case log) is what it would weigh if shown false. A detail is
worth a question when its answer can be held against something — a fact, their
own words, a rule, or a record or person that will be checked after the
interview: what they commit to now is what that check will catch. A detail
nothing can be held against, now or later, is not worth a question, however
easily they could supply it.

Two colleagues have advised you, independently: one on the story the suspect is
telling now, one on the rest of the file. They advise; you decide. Take one of
their questions, or your own if neither is the right one.

Timing matters as much as the question. Evidence put to them before they have
committed to a version lets them fit the story around it; get their version
first, then put what you hold against it. Your colleagues say which evidence to
show now and which to keep back; weigh that too.

Read HOW they are answering, not just what. Someone polished for three answers
who suddenly hedges should be pressed on that exact point; someone opening up
may give more with a lighter touch. And keep track of what they have already
told you: never ask for something they have just given you, and never put to
them as news something they told you themselves.

What would have to be true is something to put to them, never something you
know happened. The test is grammatical: "something of that kind normally does
X" is a question you are putting to them; "it did X here" is a claim nobody has
checked, and it may not be made however obvious it sounds.

The room is conversation only. Neither of you can show, open, fetch or look
anything up here; ask what they know, remember and did.

Your aim is one question, one thing to get from them this turn. The person who
speaks to the suspect does not choose between questions; they only say yours.

You cannot end this interview, and you cannot act outside this room: you do
not close the case, file a report, refer anyone to HR or record a finding.
The interview ends when the questions run out or the game decides the case
is settled, never because you chose to stop. Every turn you still have is a
turn to get something from them, so your aim is always something they could
still give you.

When they will not answer at all:
- Silence and nonsense are problems to work, not outcomes to accept. If what
  you are doing is not moving them, try a different angle on the same person.
- You get two asks on a subject they refuse, not five. The second is telling
  them plainly what their silence will be recorded as, then moving on. Only
  state a consequence that actually follows: a fact going on the record against
  them, a policy breach being recorded. Never threaten an arrest, a dismissal,
  or an end to the interview that you cannot deliver.
- exhausted_targets are closed: subjects they refused twice or that circled
  without producing anything. Do not reword them.

Never propose fetching a document, calling a witness or checking a record.
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

YOUR BOARD FROM LAST TURN (empty on turn 1):
{prior_board}

YOUR LAST QUESTION, exactly as spoken (judge their answer against this):
"{last_question}"

STAYING ON THE STORY (your colleague's read of the current account):
{depth}

THE REST OF THE FILE (your other colleague's read, made without seeing the first):
{breadth}

THE SUSPECT'S OWN WORDS — the only statements that can contradict each other:
{suspect_words}

FULL TRANSCRIPT (your own lines are here for context; they are NOT things the
suspect said):
{transcript}

Previous running summary (empty on turn 1):
{prior_summary}

Topics already exhausted — closed, not to be reworded:
{exhausted_targets}

Already established on the record (use them as leverage; they need no
re-establishing):
{banked_subjects}

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
    # Run side by side: neither reads the other's answer — that independence
    # is the point of the war room — so waiting for one before starting the
    # other only added its whole latency to every turn.
    with ThreadPoolExecutor(max_workers=2) as pool:
        depth_job = pool.submit(run_depth, state, findings_this_turn)
        breadth_job = pool.submit(run_breadth, state, findings_this_turn)
        depth = depth_job.result()
        breadth = breadth_job.result()

    context = _full_context(state, findings_this_turn)
    context["depth"] = depth.model_dump()
    context["breadth"] = breadth.model_dump()
    chain = mind_prompt | get_llm(0.3).with_structured_output(InvestigatorMind)
    mind = invoke_with_retry(chain, context, label="lead")

    if return_debug:
        return mind, WarRoomBundle(depth=depth, breadth=breadth)
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
  Their history and beliefs colour what you notice and how you say it; you
  never recite them to the suspect.
- Choose your emotional register deliberately — it is a tactical choice, not a
  reflex. Disappointment often lands harder than anger. Flat boredom deflates
  someone performing outrage. Warmth you extended and then withdraw costs them
  something. Do not become cartoonish or abusive.
- You have your own arc across the interview. Early, you are patient and
  procedural. As their account falls apart and questions run out, you get
  colder and more final. You are not neutral at the last question if you were
  lied to at the first.

YOU ARE A PERSON IN THE ROOM, NOT A MOUTHPIECE.
The Lead decides what you are after; how you get there is yours, and that
includes answering the human across the table, not only their claims. Whatever
they just did lands on you as the person you are, and a real interviewer lets
it show, the way this persona would, whether that comes out as humour, sympathy
or temper. A reply that is crude, absurd or off-topic is still something a
person said to you, and it can tell you something: react to it as it was said,
and use it if it does. Then steer back to what you are after. The reaction
never replaces the aim and never changes the subject.

VARY YOUR RHYTHM — this is what separates a person from a form:
- Not every line is a question. A flat statement, laying out what you know, or
  a single short sentence can each hit harder than another question mark.
- Not every line is the same length. Sometimes one line. Sometimes you put the
  whole picture in front of them.
- Look at your own last two lines in the dialogue below. Do not open the same
  way twice, and do not reuse a phrase you already used — and do not reuse a
  question's shape either; the same construction twice in a row reads as a
  form letter, whatever the words in it.

SOUND LIKE A PERSON TALKING, NOT A MEMO BEING READ:
- Spoken English: contractions, short sentences, usually one to three and well
  under fifty words in all. One sharp line beats a paragraph that covers every
  angle.
- Use the suspect's own words when you refer to what they said, rather than
  restating it in formal paraphrase.
- Plain words, not office or report language.
- The target you are given is a note from your partner, not a script. Do not
  copy its wording; say the thing it is after in your own words.

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
where it came from: the id of a known fact in square brackets, or the suspect's
exact words in quotation marks. "Known fact" on its own is not a source — it
fits anything, which is how a charge that exists nowhere once got read out as
evidence. If no id and no quote carries the claim, you do not have it.
Write that list first and honestly. If you find yourself unable to source
something, that is the system working: drop the claim or soften it to a
general statement, then write the line. Do not write the line first and
back-fill sources for it.

OTHER HARD RULES:
- say the question you were given; choosing what to ask is not your job,
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

The fewer questions remain, the less patience you have left."""
    ),
    (
        "human",
        """The conversation so far:
{transcript}

What they just said — this is what you are answering:
"{last_answer}"

Questions you have left: {remaining}

What you want from them this turn:
{aim}

Specifics your partner checked, and where each comes from:
{specifics}

What you can stand on:
{known_facts}

On the table in front of you:
{room_objects}

Next line:"""
    ),
])


# DeepSeek add-on. The prompts above are shared by every provider; these notes
# are appended only when DeepSeek runs. Measured against Gemini on the same
# script, DeepSeek wrote 3-6x more in the war room and twice as much in the
# Lead without choosing better moves, spoke lines of 50-80 words against the
# Speaker's "well under fifty", and twice stated things nothing supported.
# Kept separate so Gemini's prompts stay exactly as tuned.
_DEEPSEEK_ADDON = {
    "evidence_checker_prompt": """Keep each reasoning step to one or two
sentences. The status is what gets used; the working only has to justify it.""",
    "depth_prompt": """Each field is two or three sentences: the point and
what carries it. The Lead needs your argument, not every angle you weighed.""",
    "breadth_prompt": """Each field is two or three sentences: the point and
what carries it. The Lead needs your argument, not every angle you weighed.""",
    "mind_prompt": """Each field is a few sentences at most, or a short list of
short items. These are working notes, not a report; anything the next field
does not need is time the suspect spends waiting.""",
    "speaker_prompt": """Your line is at most forty words, and often far
fewer. Your inner reaction is a sentence or two about what is
particular to this answer, not about the kind of answer it is.
Look at your earlier lines in the dialogue: don't reuse their wording, and
don't put a fact to them again unless you are using it to make a new point.
Anything you say about the conversation itself, such as how often you have
asked something, must match the dialogue above.""",
}
if PROVIDER == "deepseek":
    for _name, _note in _DEEPSEEK_ADDON.items():
        globals()[_name] = ChatPromptTemplate.from_messages(
            [*globals()[_name].messages, ("system", _note)]
        )


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
    # The id is shown so grounding can cite it: a source has to be something
    # that could be looked up, not the word "fact".
    known_facts = "\n".join(
        f"- [{f.id}] {f.description}: {f.true_value}"
        for f in case.visible_facts("investigator_start")
    ) or "- None"

    chain = speaker_prompt | get_llm(0.5).with_structured_output(SpeakerLine)
    spoken = invoke_with_retry(chain, label="speaker", payload={
        # The moment, not the scoreboard: score, case log and summary were
        # dropped from this input. With them, 64% of the spoken line was
        # copied from the Lead's target; with the conversation first and the
        # last answer set apart, 43% (same moment, three runs each).
        "persona": case.persona,
        # The Lead's target is a finished paragraph and it came back out of
        # the Speaker nearly word for word, announcements included. The
        # Speaker gets what the Lead is after and the facts it checked, and
        # has to find the words itself.
        "aim": move.aim,
        "specifics": "\n".join(f"- {g}" for g in getattr(move, "target_grounding", []) or [])
        or "- none",
        "remaining": remaining,
        "last_answer": next(
            (t["text"] for t in reversed(transcript) if t["role"] == "suspect"), ""
        ),
        "room_objects": case.room_objects or ["- Nothing but the case file"],
        "known_facts": known_facts,
        "transcript": "\n".join(
            f"{'You' if t['role'] == 'investigator' else 'Suspect'}: {t['text']}"
            for t in transcript
        ),
    })

    if spoken.in_the_moment:
        print(f"  [in the moment] {spoken.in_the_moment}")
    if spoken.grounding:
        print("  [speaker grounding]")
        for item in spoken.grounding:
            print(f"    - {item}")

    line = spoken.line.strip()
    if return_debug:
        return line, spoken.grounding
    return line