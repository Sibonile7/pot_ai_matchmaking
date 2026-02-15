# AI Matchmaking for Proof of Talk (Prototype)

This prototype demonstrates an AI-powered matchmaking approach that recommends 5–8 high-value, **complementary** connections for event attendees (capital ↔ deal, bank ↔ compliant infra, buyer ↔ seller), with clear explanations and conversation starters.

## What’s included
- Sample attendee dataset (`attendees.csv`)
- Matching engine with 4-factor scoring:
  - 40% Complementarity (A wants ↔ B offers)
  - 30% Role Fit (Investor↔Founder, Bank↔Tech, etc.)
  - 20% Topic Overlap
  - 10% Novelty (avoid only obvious matches)
- Streamlit dashboard UI to view matches and “why”

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
