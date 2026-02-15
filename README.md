# AI Matchmaking for Proof of Talk (Prototype)

A lightweight AI matchmaking prototype for **Proof of Talk** that recommends **5–8 high-value, complementary** connections for event attendees (capital ↔ deal, bank ↔ compliant infra, buyer ↔ seller).  
Each recommendation includes a **match score (with factor breakdown)**, a short **“why this match”** explanation, and **conversation starters**.

## Live Links
- **Streamlit Prototype:** https://potaimatchmaking-cjhux4hc8yh4pweadh4yrk.streamlit.app/
- **Figma Wireframe:** https://www.figma.com/make/LiC9Hy2rV2pn94FGHeMVFU/Sibonile-Prototype-Figma?p=f&t=wwSHxT0YL52CV0A8-0&fullscreen=1
- **GitHub Repo:** https://github.com/Sibonile7/pot_ai_matchmaking

## What’s Included
- Fictional attendee dataset: `attendees.csv` (10 profiles)
- Matching engine: `matching.py`
  - **40% Complementarity (two-way):** 0.5·sim(A wants, B offers) + 0.5·sim(B wants, A offers)
  - **30% Role Fit:** Investor↔Founder, Operator↔Founder, Investor↔Operator, etc.
  - **20% Topic Overlap**
  - **10% Novelty** (avoids only obvious matches)
- Streamlit app UI: `app.py`
  - Select an attendee and view top matches
  - Score + breakdown + explanations + conversation starters
  - Export match results as CSV
- Optional external retrieval (public web): `company_url` → website snippet shown in sidebar (no LinkedIn scraping)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
