import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROLE_COMPAT: Dict[Tuple[str, str], float] = {
    ("Investor", "Founder"): 1.0,
    ("Founder", "Investor"): 1.0,
    ("Operator", "Founder"): 0.8,
    ("Founder", "Operator"): 0.8,
    ("Operator", "Investor"): 0.6,
    ("Investor", "Operator"): 0.6,
    ("Investor", "Investor"): 0.4,
    ("Founder", "Founder"): 0.4,
    ("Operator", "Operator"): 0.3,
}

THEME_KEYWORDS = [
    "rwa", "tokenization", "custody", "compliance", "defi", "zk", "layer2",
    "identity", "kyc", "bank", "institutional", "exchange", "privatecredit"
]

SYNONYMS = {
    r"\binstitutional investors?\b": "capital",
    r"\binvestors?\b": "capital",
    r"\blps?\b": "capital",
    r"\bcapital allocation\b": "capital",
    r"\bfundraising\b": "raise capital",
    r"\bcustody\b": "custody compliance",
    r"\bregulated\b": "compliance",
    r"\bkyc\b": "identity compliance",
    r"\btokenization\b": "rwa tokenization",
    r"\bprivate credit\b": "privatecredit",
    r"\bdefi\b": "defi yield",
    r"\bdistribution\b": "clients distribution",
}

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    for pat, rep in SYNONYMS.items():
        s = re.sub(pat, rep, s)
    return s


def _norm_tags(tags: str) -> List[str]:
    if not isinstance(tags, str):
        return []
    return [t.strip().lower() for t in tags.split(",") if t.strip()]


def role_fit(a_type: str, b_type: str) -> float:
    return ROLE_COMPAT.get((a_type, b_type), 0.4)


def topic_overlap(a_tags: List[str], b_tags: List[str]) -> float:
    a = set(a_tags)
    b = set(b_tags)
    overlap = len(a.intersection(b))
    # Convert overlap count to [0..1] roughly; cap so it doesn’t dominate
    return min(1.0, overlap / 6.0)


def novelty_boost(a_tags: List[str], b_tags: List[str], sim: float) -> float:
    """
    Novelty: reward non-obvious matches slightly.
    If text similarity is very high, novelty is lower.
    If similarity is moderate but role fit is good, novelty can help.
    """
    shared = set(a_tags).intersection(set(b_tags))
    shared_themes = [t for t in shared if t in THEME_KEYWORDS]
    theme_factor = 1.0 - min(1.0, len(shared_themes) / 5.0)  # fewer shared themes => more novelty
    sim_factor = 1.0 - sim  # lower similarity => more novelty
    return float(np.clip(0.6 * theme_factor + 0.4 * sim_factor, 0.0, 1.0))


def build_text_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Complementarity: compare A "looking_for" against B "offers"
    want_raw = (df["looking_for"].fillna("") + " " + df["focus"].fillna("") + " " + df["tags"].fillna(""))
    offer_raw = (df["offers"].fillna("") + " " + df["focus"].fillna("") + " " + df["tags"].fillna(""))

    want_text = want_raw.apply(normalize_text)
    offer_text = offer_raw.apply(normalize_text)
    return want_text, offer_text


def generate_starters(a_row: pd.Series, b_row: pd.Series) -> List[str]:
    """
    Role-aware + theme-aware conversation starters (no LLM needed).
    Produces less repetitive, more "deal-like" questions.
    """
    a_type = str(a_row.get("type", "")).strip()
    b_type = str(b_row.get("type", "")).strip()

    a_tags = _norm_tags(a_row.get("tags", ""))
    b_tags = _norm_tags(b_row.get("tags", ""))

    shared = [t for t in set(a_tags).intersection(set(b_tags)) if t in THEME_KEYWORDS]
    theme = shared[0] if shared else None

    starters: List[str] = []

    # 1) Theme opener (if available)
    if theme:
        starters.append(f"What’s your current focus in {theme.upper()}—and what would success look like by end of 2026?")
    else:
        starters.append("What problem are you most motivated to solve during Proof of Talk?")

    # 2) Role-aware "deal" question
    pair = (a_type, b_type)
    if pair in [("Investor", "Founder"), ("Founder", "Investor")]:
        starters.append("If we explored an investment, what are the top 2 metrics/traction signals that matter most right now?")
    elif pair in [("Operator", "Founder"), ("Founder", "Operator")]:
        starters.append("What would a realistic integration or pilot look like in the next 30–60 days (systems, data, compliance)?")
    elif pair in [("Operator", "Investor"), ("Investor", "Operator")]:
        starters.append("Where do you see the biggest bottleneck for institutional adoption—and what kind of partner would remove it?")
    else:
        starters.append("What’s the most valuable partnership you could form at this event, and why?")

    # 3) Close with a concrete next step
    starters.append("If this meeting goes well, what is the one next step we should agree on before we leave the room?")

    return starters[:3]


def compute_matches(df: pd.DataFrame, idx: int, k: int = 8) -> List[Dict]:
    want_text, offer_text = build_text_fields(df)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    # Fit on both want+offer text so vocabulary covers both directions
    all_text = pd.concat([want_text, offer_text], ignore_index=True)
    vectorizer.fit(all_text)

    V_offer = vectorizer.transform(offer_text)   # each person's offers
    V_want  = vectorizer.transform(want_text)    # each person's wants

    a = df.iloc[idx]
    a_tags = _norm_tags(a.get("tags", ""))

    a_want = V_want[idx]     # A wants
    a_offer = V_offer[idx]   # A offers

    # Direction 1: A wants vs B offers
    comp_a_to_b = cosine_similarity(a_want, V_offer).flatten()

    # Direction 2: B wants vs A offers
    comp_b_to_a = cosine_similarity(V_want, a_offer).flatten()

    # Bidirectional complementarity (average both directions)
    comp_sim = 0.5 * comp_a_to_b + 0.5 * comp_b_to_a

    results: List[Dict] = []
    for j in range(len(df)):
        if j == idx:
            continue

        b = df.iloc[j]
        b_tags = _norm_tags(b.get("tags", ""))

        # Factors (0..1)
        comp = float(np.clip(comp_sim[j], 0.0, 1.0))
        rf = float(role_fit(str(a.get("type", "")), str(b.get("type", ""))))
        to = float(topic_overlap(a_tags, b_tags))
        nov = float(novelty_boost(a_tags, b_tags, comp))

        # Final weighted score
        score = 0.40 * comp + 0.30 * rf + 0.20 * to + 0.10 * nov

        # Explanations
        why: List[str] = []
        why.append(f"Complementarity (two-way): mutual want↔offer alignment ({comp*100:.0f}%).")
        why.append(f"Role fit: {a['type']} ↔ {b['type']} ({rf*100:.0f}%).")


        shared = sorted(list(set(a_tags).intersection(set(b_tags))))
        if shared:
            why.append("Shared themes: " + ", ".join(shared[:4]) + ".")
        else:
            why.append("Novel connection: low overlap but strong counterparty fit.")

        results.append({
            "match_index": j,
            "score": float(score),
            "comp": comp,
            "role_fit": rf,
            "topic": to,
            "novelty": nov,
            "why": why,
            "starters": generate_starters(a, b),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
