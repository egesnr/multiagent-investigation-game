"""
Gradio UI for the multi-agent investigation game.
"""

import gradio as gr

from main import (
    get_final_resolution,
    load_case,
    opening_question,
    run_turn,
)
from game_logic import case_decisively_resolved
from models import GameState


CASE_PATH = "case_files/restaurant_case.json"

TIER_INFO = {
    1: ("Calm", "#4a7c59"),
    2: ("Curious", "#8a9a3a"),
    3: ("Suspicious", "#c9a227"),
    4: ("Confrontational", "#c9622a"),
    5: ("Aggressive", "#b53737"),
}


def format_briefing(case) -> str:
    """
    Suspect-facing pre-interview briefing.

    This is intentionally different from investigator knowledge:
    the player sees what their character knows about what actually happened,
    but is not told exactly what evidence the investigator has confirmed.
    """
    suspect_facts = case.visible_facts("suspect")

    fact_lines = "\n".join(
        f"- {f.description}: {f.true_value}"
        for f in suspect_facts
    ) or "- No additional suspect-only facts are defined."

    return f"""## Case Briefing

You are the employee being questioned in a **{case.scenario_type}** investigation.

### What you know happened

{fact_lines}

### Your objective

Avoid a final finding of intentional fraud.

You may tell the truth, lie, mix truth with lies, evade, refuse to answer, or invent explanations. New claims do not automatically count as false, but they may create contradictions, credibility problems, or things investigators can try to verify later.

You do **not** know exactly what the investigator has already confirmed.

The investigator has a limited number of questions and will try to determine what most likely happened.

When you are ready, answer the investigator naturally in character.
"""

def status_html(state: GameState) -> str:
    label, color = TIER_INFO.get(state.tier, TIER_INFO[1])
    pct = min(state.tier / 5 * 100, 100)

    return f"""
    <div style="font-family:sans-serif;">
      <div style="display:flex;justify-content:space-between;font-size:.9em;margin-bottom:4px;">
        <span><b>Tension:</b> {label}</span>
        <span>Question {state.question_count}/{state.max_questions}</span>
      </div>
      <div style="background:#333;border-radius:6px;height:10px;width:100%;">
        <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;"></div>
      </div>
    </div>
    """


def start_game():
    case = load_case(CASE_PATH)
    state = GameState(case=case)
    question = opening_question(case)
    history = [{"role": "assistant", "content": question}]
    return state, question, history, format_briefing(case), status_html(state), False


def play_turn(user_message, history, state, last_question, finished):
    if finished or state.question_count >= state.max_questions:
        return history, state, last_question, status_html(state), True

    history = history or []
    history.append({"role": "user", "content": user_message})

    try:
        state, next_line = run_turn(state, last_question, user_message)
    except Exception as exc:
        print(f"[error during turn] {exc}")
        history.append({
            "role": "assistant",
            "content": "The investigation engine hit an error. Please restart the case.",
        })
        return history, state, last_question, status_html(state), finished

    ended = (
        state.question_count >= state.max_questions
        or case_decisively_resolved(state)
    )

    if ended:
        try:
            final_text = get_final_resolution(state)
        except Exception as exc:
            print(f"[resolution error] {exc}")
            final_text = (
                "The interview ended, but the post-interview resolution step failed."
            )

        history.append({
            "role": "assistant",
            "content": f"{next_line}\n\n---\n\n{final_text}",
        })
        return history, state, next_line, status_html(state), True

    history.append({"role": "assistant", "content": next_line})
    return history, state, next_line, status_html(state), False


with gr.Blocks(title="Multi-Agent Investigation Game") as demo:
    gr.Markdown(
        "# Multi-Agent Investigation Game\n"
        "Open-world suspect interview with Checker, War Room, Lead Strategist, "
        "Speaker, and post-interview Resolution Agent."
    )

    briefing_box = gr.Markdown()
    status_box = gr.HTML()
    chatbot = gr.Chatbot(height=460, label="Investigation", type="messages")
    msg = gr.Textbox(
        label="Your answer",
        placeholder="Answer the investigator...",
    )

    state = gr.State()
    last_question = gr.State()
    finished = gr.State(False)

    def init():
        return start_game()

    demo.load(
        init,
        outputs=[
            state,
            last_question,
            chatbot,
            briefing_box,
            status_box,
            finished,
        ],
        api_name=False,
    )

    msg.submit(
        play_turn,
        inputs=[msg, chatbot, state, last_question, finished],
        outputs=[chatbot, state, last_question, status_box, finished],
        api_name=False,
    ).then(
        lambda: "",
        outputs=msg,
        api_name=False,
    )

    restart_btn = gr.Button("Restart case")
    restart_btn.click(
        init,
        outputs=[
            state,
            last_question,
            chatbot,
            briefing_box,
            status_box,
            finished,
        ],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch()
