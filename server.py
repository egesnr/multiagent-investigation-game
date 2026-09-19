"""
FastAPI backend for "The Box" — the browser UI in web/index.html.

Per-browser-session in memory (a cookie keys into a process-local dict) —
enough for a handful of people to play without stomping on each other's
game, but nothing here is durable: a redeploy or a free-tier host's idle
sleep wipes every in-progress interview. That's an accepted trade-off for a
demo of the agent architecture, not a hosted product; adding real persistence
would be a different, much bigger project.

Replaces app.py's Gradio UI, which couldn't reliably run the room's typing
animation, dactilo sound, and continuous camera effects (Gradio's HTML
component doesn't re-run injected <script> tags on update).

Deliberately reads GameState/CaseLog as they already exist — no new fields,
no changes to main.py/models.py/agents.py/game_logic.py. The tactic/demeanor
label systems the UI was originally built against were removed from the
engine (see commit 34e228d) for good, measured reasons; this does not
reintroduce anything like them, even for display purposes.
"""

import logging
import uuid
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main import DEFAULT_CASE_PATH, load_case, opening_question, run_turn
from game_logic import IMPACT_POINTS, case_decisively_resolved
from models import ClaimType, FindingBasis, GameState
from resolution import run_resolution


WEB_DIR = Path(__file__).parent / "web"
OBJECTIVE_TEXT = "Avoid a final finding of intentional fraud."
SESSION_COOKIE = "the_box_session"

app = FastAPI()

# session_id -> {"state": GameState, "last_question": str}. Process-local,
# not shared across workers/replicas — this app must run as a single process.
_sessions: dict[str, dict] = {}


class TurnRequest(BaseModel):
    answer: str


def _get_session(request: Request) -> dict:
    session_id = request.cookies.get(SESSION_COOKIE)
    session = _sessions.get(session_id) if session_id else None
    if session is None:
        raise HTTPException(400, "No active interview — call /api/start first.")
    return session


def _persona_line(persona: str) -> str:
    """The descriptive clause before the first colon, e.g. 'Senior Corporate
    Fraud Auditor named Dana Whitfield' — works for any case file's persona
    text without hardcoding a name-extraction pattern."""
    return persona.split(":", 1)[0].strip()


def _finding_tag(claim, state: GameState) -> str:
    if claim.claim_type == ClaimType.CONTRADICTION:
        return "CONTRADICTION"
    if claim.claim_type == ClaimType.ADMISSION:
        return "ADMISSION"
    if claim.basis == FindingBasis.POLICY:
        return "POLICY BREACH"
    if claim.text in state.case_log.evasions:
        return "EVASION"
    return claim.claim_type.value.upper()


def _turn_findings(state: GameState) -> list[dict]:
    """This turn's newly-logged, investigator-visible claims that actually
    carried scoring weight — read straight from case_log.claims (each one
    already carries the turn it was logged on), no engine changes needed."""
    findings = []
    for claim in state.case_log.claims:
        if claim.turn != state.question_count or not claim.investigator_visible:
            continue
        points = IMPACT_POINTS.get(claim.evidentiary_impact, 0)
        if points <= 0:
            continue
        findings.append({"tag": _finding_tag(claim, state), "points": points})
    return findings


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/start")
def start(response: Response):
    case = load_case(DEFAULT_CASE_PATH)
    state = GameState(case=case)
    question = opening_question(case)

    session_id = uuid.uuid4().hex
    _sessions[session_id] = {"state": state, "last_question": question}
    response.set_cookie(
        SESSION_COOKIE, session_id, httponly=True, samesite="lax", max_age=6 * 3600
    )

    suspect_facts = [
        f"{f.description}: {f.true_value}" for f in case.visible_facts("suspect")
    ]
    present_people = [
        p for p in case.present_people if p not in ("suspect", "investigator")
    ]

    return {
        "scenario_type": case.scenario_type,
        "persona_line": _persona_line(case.persona),
        "suspect_facts": suspect_facts,
        "room_objects": case.room_objects,
        "present_people": present_people,
        "objective": OBJECTIVE_TEXT,
        "opening_question": question,
        "max_questions": state.max_questions,
        "arrest_threshold": case.arrest_threshold,
    }


@app.post("/api/turn")
def turn(payload: TurnRequest, request: Request):
    session = _get_session(request)
    state = session["state"]
    last_question = session["last_question"]

    before_count = state.question_count
    try:
        state, next_line = run_turn(state, last_question, payload.answer)
    except Exception as exc:
        # A turn runs ~7 sequential LLM calls; llm_utils already retries
        # transient failures internally, but anything that still fails after
        # all retries has no better move than to say so plainly and leave
        # state untouched — the question wasn't consumed. Log the real
        # exception (type + message) so a misconfigured key or a genuine
        # rate limit can be told apart from Render's logs, instead of both
        # showing the same generic message to the player.
        logger.error("Turn failed: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            503,
            "The investigator's line is jammed right now (the model API "
            f"raised {type(exc).__name__}) — wait a moment and send your "
            "answer again.",
        )
    session["state"] = state

    # A pure blocked world action doesn't consume the question — the same
    # question stands and nothing else in the room state changes. See
    # main.run_turn and app.py's equivalent check.
    if state.question_count == before_count and state.last_world_notice:
        return {
            "blocked": True,
            "notice": state.last_world_notice,
            "question_count": state.question_count,
            "max_questions": state.max_questions,
        }

    session["last_question"] = next_line

    ended = (
        state.question_count >= state.max_questions
        or case_decisively_resolved(state)
    )

    response = {
        "blocked": False,
        "question_count": state.question_count,
        "max_questions": state.max_questions,
        "score": state.score,
        "tier": state.tier,
        "thread": state.last_move,
        "next_line": next_line,
        "findings": _turn_findings(state),
        "ended": ended,
        "resolution": None,
    }

    if ended:
        report = run_resolution(state)
        response["resolution"] = {
            "outcome": report.outcome.value,
            "aftermath": report.aftermath,
            "final_score": report.final_score,
            "threshold": state.case.arrest_threshold,
        }
        del _sessions[request.cookies[SESSION_COOKIE]]

    return response


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
