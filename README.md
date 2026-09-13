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

## Important design boundaries

- Authored world truth and investigator knowledge are separate.
- The Checker may see full truth, but hidden-truth-only findings are not exposed to the War Room or Speaker.
- Unknown does not mean false.
- New player stories can become unresolved claims without needing to be pre-authored in JSON.
- The Strategist has a limited question budget and should prioritize information value.
- The Resolution Agent may resolve claims only from authored facts; it must not fabricate decisive evidence.
- The hidden `guilty` field is a benchmark label, not evidence.

## Run

```bash
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key_here
set GAME_MODEL=gemini-3.1-flash-lite
python main.py
```

Or:

```bash
python app.py
```
