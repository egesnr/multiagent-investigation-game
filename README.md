---
title: The Box
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Agent Investigation Game

An open-world suspect interview used to demonstrate and stress-test a multi-agent investigation architecture.

## Interview architecture

Player answer  
→ Extractor  
→ Checker  
→ Shared Context  
→ internal War Room:
- Evidence Analyst
- Skeptical Investigator
- Alternative-Hypothesis Investigator  
→ Lead Strategist  
→ Speaker

After the interview:
→ Resolution Agent  
→ final outcome

The War Room is good cop / bad cop by design: the Skeptical Investigator always argues for pressing harder, the Alternative-Hypothesis Investigator always argues the innocent reading, and the Lead Strategist has to pick a side each turn rather than split the difference.

## What this is

Most interrogation games are dialogue trees: a fixed set of questions, a fixed set of suspect responses, and a script that only ever plays out the branches someone authored in advance. This is the opposite bet — the player can answer *anything*, in their own words, and the game has to figure out in real time whether that story holds up, using a multi-agent pipeline instead of a script to do the figuring.

## How to play

You play the suspect. The AI plays the investigator, built from a case file that authors a hidden ground truth — what actually happened, who could plausibly know it, and what counts as damaging.

- You're given a briefing on what your character actually knows, and an objective (usually: avoid being found guilty).
- Each round, the investigator asks a question and you type a free-text answer. You can tell the truth, lie, evade, refuse, or invent a story — nothing you type has to match a preset option.
- Behind the scenes, the pipeline pulls out what you just claimed, checks it against the hidden facts and against everything you've said before, and decides how damaging it is — then the investigator picks its next question based on that, the same way a real interviewer follows up on a weak spot instead of asking down a fixed list.
- You don't know exactly what the investigator has already confirmed, so you're guessing how much room you have.
- The interview ends after a limited number of questions (or sooner, if the investigator decides they have enough), and a final resolution is produced from the facts that were actually established — never from something invented on the spot just to close the case out neatly.

## Run

```bash
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key_here
set GAME_MODEL=gemini-3.1-flash-lite
python main.py
```

Or, for the browser UI ("The Box" — an animated interrogation room instead of a terminal transcript):

```bash
python server.py
```

Then open http://127.0.0.1:8000. It's a single FastAPI process serving `web/index.html` and running the same pipeline as `main.py` underneath — a real interview, so each answer takes a few seconds to get a reply. Single session at a time by design; it's a demo of the agent architecture, not a hosted multi-user service.
