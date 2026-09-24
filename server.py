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
import threading
import time
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

# job_id -> {"status": "pending" | "done" | "error", "result"/"detail": ...}.
# A turn runs ~7 sequential LLM calls and can genuinely take over a minute —
# longer than some hosts' front-end proxy will hold a request open (observed
# on Render: the connection gets dropped well before the backend finishes,
# even though the backend itself never errors). So /api/turn doesn't block:
# it starts the work in a background thread and returns a job id right away;
# the frontend polls /api/turn/{job_id} until it's done. Each individual
# HTTP request completes in well under a second either way, so no proxy's
# timeout — known or not — is ever in play.
_jobs: dict[str, dict] = {}


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


def _case_findings(state: GameState) -> list[dict]:
    """Every scoring, investigator-visible claim from the whole interview —
    the same set the evidence board pinned turn by turn, with its text, for
    the resolution screen's account of why the case ended the way it did."""
    findings = []
    for claim in state.case_log.claims:
        if not claim.investigator_visible:
            continue
        points = IMPACT_POINTS.get(claim.evidentiary_impact, 0)
        if points <= 0:
            continue
        findings.append({
            "tag": _finding_tag(claim, state),
            "text": claim.text,
            "turn": claim.turn,
            "points": points,
        })
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


def _process_turn(session_id: str, session: dict, answer: str, job_id: str) -> None:
    """Runs the real turn (all ~7 LLM calls) on a background thread. Never
    raises — any failure is recorded on the job instead, since there's no
    HTTP request left by the time this finishes to raise an exception into."""
    state = session["state"]
    last_question = session["last_question"]
    before_count = state.question_count

    try:
        state, next_line = run_turn(state, last_question, answer)
    except Exception as exc:
        # llm_utils already retries transient failures internally; anything
        # that still fails after all retries has no better move than to say
        # so plainly and leave state untouched — the question wasn't
        # consumed. Log the real exception (type + message) so a
        # misconfigured key or a genuine rate limit can be told apart from
        # Render's logs, instead of both showing the same generic message.
        logger.error("Turn failed: %s: %s", type(exc).__name__, exc)
        _jobs[job_id] = {
            "status": "error",
            "detail": (
                "The investigator's line is jammed right now (the model "
                f"API raised {type(exc).__name__}) — wait a moment and "
                "send your answer again."
            ),
        }
        return

    session["state"] = state

    # A pure blocked world action doesn't consume the question — the same
    # question stands and nothing else in the room state changes. See
    # main.run_turn and app.py's equivalent check.
    if state.question_count == before_count and state.last_world_notice:
        _jobs[job_id] = {
            "status": "done",
            "result": {
                "blocked": True,
                "notice": state.last_world_notice,
                "question_count": state.question_count,
                "max_questions": state.max_questions,
            },
        }
        return

    session["last_question"] = next_line

    ended = (
        state.question_count >= state.max_questions
        or case_decisively_resolved(state)
    )

    result = {
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
        result["resolution"] = {
            "outcome": report.outcome.value,
            "aftermath": report.aftermath,
            "reasoning": report.reasoning,
            "confidence": report.confidence,
            "interview_score": state.score,
            "verification_score_delta": report.verification_score_delta,
            "final_score": report.final_score,
            "threshold": state.case.arrest_threshold,
            "findings": _case_findings(state),
            "verifications": [
                {
                    "claim": v.claim,
                    "status": v.status.value,
                    "basis": v.basis,
                    "points": IMPACT_POINTS.get(v.evidentiary_impact, 0),
                }
                for v in report.verifications
            ],
        }
        del _sessions[session_id]

    _jobs[job_id] = {"status": "done", "result": result}


# job_id -> [started, finished]. Measures what the engine's own timing cannot:
# how long a finished answer sat waiting for the browser's next poll.
_job_clock: dict[str, list] = {}


def _timed_process_turn(session_id: str, session: dict, answer: str, job_id: str) -> None:
    _process_turn(session_id, session, answer, job_id)
    if job_id in _job_clock:
        _job_clock[job_id][1] = time.perf_counter()


@app.post("/api/turn")
def turn(payload: TurnRequest, request: Request):
    session_id = request.cookies.get(SESSION_COOKIE)
    session = _get_session(request)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending"}
    _job_clock[job_id] = [time.perf_counter(), None]
    threading.Thread(
        target=_timed_process_turn,
        args=(session_id, session, payload.answer, job_id),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/turn/{job_id}")
def turn_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or already-collected job.")

    if job["status"] == "pending":
        return {"status": "pending"}

    del _jobs[job_id]
    started, finished = _job_clock.pop(job_id, [None, None])
    if started is not None:
        now = time.perf_counter()
        finished = finished or now
        logger.info(
            "[timing] engine %.1fs + waiting for browser poll %.1fs = %.1fs until the answer left the server",
            finished - started, now - finished, now - started,
        )
    if job["status"] == "error":
        raise HTTPException(503, job["detail"])
    return job["result"]


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
