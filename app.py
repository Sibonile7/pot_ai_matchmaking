import pandas as pd
import streamlit as st
from matching import compute_matches
import requests
from bs4 import BeautifulSoup


# ----------------------------
# Page config MUST be first
# ----------------------------
st.set_page_config(page_title="Proof of Talk — AI Matchmaking (Prototype)", layout="wide")

st.info(
    "Prototype demo uses fictional attendee data. No external scraping. "
    "Optional enrichment pulls only public website text. Production version supports opt-in enrichment."
)


# ----------------------------
# Data loading + enrichment
# ----------------------------
@st.cache_data
def load_data(path: str = "attendees.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure expected columns exist
    for c in ["focus", "looking_for", "offers", "tags", "company_url"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    return df


@st.cache_data(show_spinner=False)
def fetch_company_snippet(url: str) -> str:
    """
    Level 3 external retrieval: fetch a short public snippet from a URL (title + meta description + first paragraphs).
    Keep it lightweight, capped for UI.
    """
    if not isinstance(url, str) or not url.strip():
        return ""

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url.strip(), headers=headers, timeout=8)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        title = (soup.title.text.strip() if soup.title else "")
        meta = soup.find("meta", attrs={"name": "description"})
        desc = (meta.get("content", "").strip() if meta else "")

        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        body = " ".join(paras[:2]).strip()

        snippet = " ".join([t for t in [title, desc, body] if t])
        return snippet[:500]
    except Exception:
        return ""


# ----------------------------
# App UI
# ----------------------------
st.title("AI Matchmaking for Proof of Talk")
st.caption("Prototype: complementary matching (capital ↔ deal, bank ↔ compliant infra, buyer ↔ seller) with explanations.")

df = load_data()

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Select Attendee")

    options = [f"{r['name']} — {r['role']} ({r['org']})" for _, r in df.iterrows()]
    choice = st.selectbox("Attendee", options)
    idx = options.index(choice)
    a = df.iloc[idx]

    st.markdown("### Profile")
    st.write(f"**Type:** {a['type']}")
    st.write(f"**Focus:** {a['focus']}")
    st.write(f"**Looking for:** {a['looking_for']}")
    st.write(f"**Offers:** {a['offers']}")
    st.write(f"**Tags:** {a['tags']}")

    # External enrichment (Level 3)
    url = a.get("company_url", "")
    if isinstance(url, str) and url.strip():
        with st.expander("Company Enrichment (public web)", expanded=False):
            st.caption(url)
            snippet = fetch_company_snippet(url)
            st.write(snippet if snippet else "Could not fetch website snippet (demo).")

    k = st.slider("Number of matches to show", 3, 10, 8)

with right:
    st.subheader(f"Top {k} Matches (with reasons)")
    matches = compute_matches(df, idx, k=k)

    # Optional: export matches as CSV (organizer-friendly)
    export_rows = []
    for m in matches:
        b = df.iloc[m["match_index"]]
        export_rows.append({
            "attendee": a["name"],
            "match": b["name"],
            "match_role": b["role"],
            "match_org": b["org"],
            "match_type": b["type"],
            "score": round(float(m["score"]), 3),
        })
    export_df = pd.DataFrame(export_rows)
    st.download_button(
        label="Export these matches (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"matches_{a['name'].replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    for m in matches:
        b = df.iloc[m["match_index"]]
        with st.container(border=True):
            st.markdown(f"### {b['name']}")
            st.write(f"**{b['role']}**, {b['org']}  \nType: {b['type']}")

            st.progress(min(1.0, float(m["score"])))
            st.caption(f"Match score: {float(m['score'])*100:.0f}%")

            # If you added breakdown fields in matching.py, show them (won't crash if missing)
            if all(k_ in m for k_ in ["comp", "role_fit", "topic", "novelty"]):
                st.caption(
                    f"Breakdown — Complementarity: {m['comp']*100:.0f}% • "
                    f"Role fit: {m['role_fit']*100:.0f}% • "
                    f"Topic: {m['topic']*100:.0f}% • "
                    f"Novelty: {m['novelty']*100:.0f}%"
                )

               
            st.markdown("**Why this match?**")
            for line in m["why"]:
                st.write(f"- {line}")

            st.markdown("**Conversation starters**")
            for s in m["starters"]:
                st.write(f"- {s}")

            col1, col2 = st.columns(2)
            with col1:
                st.button("Request Introduction", key=f"intro_{a['name']}_{b['name']}")
            with col2:
                st.button("Save for Later", key=f"save_{a['name']}_{b['name']}")
