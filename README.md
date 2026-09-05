# Multi-Agent Police Investigation Game

An open-world investigation game powered by multiple AI agents.

The player acts as an investigator interviewing a suspect, identifying contradictions, following leads, requesting evidence, and deciding whether the case is strong enough to support an arrest.

The system separates **hidden world truth** from **investigator-visible evidence**, allowing suspects to lie, remain ambiguous, or reveal information gradually without leaking the authored solution to the player-facing agents.

## Architecture

The game uses several cooperating agents with different responsibilities:

* **Strategist / War Room** — decides which investigative thread to pursue next.
* **Speaker** — conducts the interview and generates the suspect-facing interaction.
* **Checker** — evaluates claims, contradictions, evidence, and credibility.
* **Resolution Agent** — performs final verification and determines the case outcome.

A central `GameState` stores the shared investigation state across turns.

## Core Design Principle

The authored `CaseFile` contains the fixed truth of the scenario.

Hidden facts may be accessed by verification and resolution logic, but they must not leak into the investigator-facing context unless the investigator has actually established them through questioning or evidence.

This separation prevents the interview agents from accidentally knowing the answer in advance.

## Project Structure

```text
multiagent-investigation-game/
├── agents.py
├── app.py
├── game_logic.py
├── main.py
├── models.py
├── resolution.py
├── smoke_test.py
├── requirements.txt
├── assets/
│   └── investigator.png
└── case_files/
    └── restaurant_case.json
```

## Core Data Model

### CaseFile

A `CaseFile` defines the objective world state of a scenario.

```python
class CaseFile(BaseModel):
    scenario_type: str
    persona: str
    guilty: bool
    facts: list[Fact]
    policy_rules: list[str] = Field(default_factory=list)
    arrest_threshold: int = 80
    claim_categories: list[str] = Field(default_factory=list)
```

Each fact can specify which actors are allowed to see it.

```python
def visible_facts(self, actor: str) -> list[Fact]:
    return [f for f in self.facts if actor in f.visible_to]
```

### Case Log

The `CaseLog` tracks what has actually been established during the investigation.

It includes:

* claims made by the suspect
* open investigation leads
* contradictions
* admissions
* credibility issues
* evidence requests
* unresolved claims
* exhausted investigative targets

The investigator-facing view explicitly removes hidden-truth findings.

```python
def investigator_view(self) -> dict:
    """Safe context for investigator-facing agents."""
```

### GameState

`GameState` represents the evolving investigation.

It contains:

```python
class GameState(BaseModel):
    case: CaseFile
    transcript: list[dict]
    case_log: CaseLog

    score: int = 0
    audit_debt: int = 0
    tier: int = 1
    question_count: int = 0
    max_questions: int = 7
```

It also tracks flagged facts, previous moves, and investigation history.

## Investigation Flow

A typical turn follows this structure:

```text
Investigator question
        ↓
Strategist chooses investigative target
        ↓
Speaker generates response
        ↓
Checker evaluates claims and contradictions
        ↓
CaseLog / GameState updated
        ↓
Next investigation turn
```

At the end of the investigation, the Resolution Agent evaluates the accumulated evidence.

Possible outcomes include:

```text
CAUGHT
NOT_PROVEN
POLICY_VIOLATION_ONLY
```

## Evidence and Verification

Claims can be classified as:

```text
SUPPORTED
CONTRADICTED
UNVERIFIED
ADMITTED
```

Verification results can be:

```text
CONFIRMED
DISPROVED
INCONCLUSIVE
NOT_MATERIAL
```

This distinction allows the game to model uncertainty instead of treating every statement as simply true or false.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Or, depending on the entry point:

```bash
python main.py
```

## Security

Local configuration and credentials should not be committed to Git.

The repository should exclude files such as:

```text
.env
.gradio/
*.pem
*.key
```

Use environment variables for API credentials.

## Status

This project is currently under active development.

Current work includes improving:

* multi-agent interview behavior
* contradiction detection
* evidence verification
* investigation strategy
* final case resolution
* game balancing and scoring
