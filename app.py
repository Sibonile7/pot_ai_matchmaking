import pandas as pd
import streamlit as st
from matching import compute_matches

st.set_page_config(page_title="Proof of Talk — AI Matchmaking (Prototype)", layout="wide")

@st.cache_data
def load_data(path="attendees.csv"):
    df = pd.read_csv(path)
    for c in ["focus", "looking_for", "offers", "tags"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    return df

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

    k = st.slider("How many matches?", 5, 8, 8)

with right:
    st.subheader(f"Top {k} Matches (with reasons)")
    matches = compute_matches(df, idx, k=k)

    for m in matches:
        b = df.iloc[m["match_index"]]
        with st.container(border=True):
            st.markdown(f"### {b['name']}")
            st.write(f"**{b['role']}**, {b['org']}  \nType: {b['type']}")
            st.progress(min(1.0, m["score"]))
            st.caption(f"Match score: {m['score']:.2f}")

            st.markdown("**Why this match?**")
            for line in m["why"]:
                st.write(f"- {line}")

            st.markdown("**Conversation starters**")
            for s in m["starters"]:
                st.write(f"- {s}")

            col1, col2 = st.columns(2)
            with col1:
                st.button("Request Introduction", key=f"intro_{b['name']}")
            with col2:
                st.button("Save for Later", key=f"save_{b['name']}")
