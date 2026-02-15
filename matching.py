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
    want_text = (df["looking_for"].fillna("") + " " + df["focus"].fillna("") + " " + df["tags"].fillna("")).str.lower()
    offer_text = (df["offers"].fillna("") + " " + df["focus"].fillna("") + " " + df["tags"].fillna("")).str.lower()
    return want_text, offer_text


def generate_starters(a_row: pd.Series, b_row: pd.Series) -> List[str]:
    """
    Simple, deterministic conversation starters (no LLM needed).
    """
    starters = []
    a_tags = _norm_tags(a_row.get("tags", ""))
    b_tags = _norm_tags(b_row.get("tags", ""))
    shared = [t for t in set(a_tags).intersection(set(b_tags)) if t in THEME_KEYWORDS]
    if shared:
        starters.append(f"How are you approaching {shared[0].upper()} in your current work?")
    starters.append("What’s your #1 goal you want to achieve at Proof of Talk?")
    starters.append("What would a successful partnership look like in the next 90 days?")
    return starters[:3]


def compute_matches(df: pd.DataFrame, idx: int, k: int = 8) -> List[Dict]:
    want_text, offer_text = build_text_fields(df)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    # Fit on both want+offer text so vocabulary covers both directions
    all_text = pd.concat([want_text, offer_text], ignore_index=True)
    vectorizer.fit(all_text)

    V_offer = vectorizer.transform(offer_text)   # each person's offers
    V_want  = vectorizer.transform(want_text)    # each person's wants

    a_want = V_want[idx]     # A wants
    a_offer = V_offer[idx]   # A offers

    # Direction 1: A wants vs B offers
    comp_a_to_b = cosine_similarity(a_want, V_offer).flatten()

    # Direction 2: B wants vs A offers
    comp_b_to_a = cosine_similarity(V_want, a_offer).flatten()

    # Bidirectional complementarity
    comp_sim = 0.5 * comp_a_to_b + 0.5 * comp_b_to_a


    a = df.iloc[idx]
    a_tags = _norm_tags(a.get("tags", ""))

    results = []
    for j in range(len(df)):
        if j == idx:
            continue
        b = df.iloc[j]
        b_tags = _norm_tags(b.get("tags", ""))

        # Your weights:
        # 40% complementarity + 30% role fit + 20% topic overlap + 10% novelty
        comp = float(np.clip(comp_sim[j], 0.0, 1.0))
        rf = role_fit(a["type"], b["type"])
        to = topic_overlap(a_tags, b_tags)
        nov = novelty_boost(a_tags, b_tags, comp)

        score = 0.40 * comp + 0.30 * rf + 0.20 * to + 0.10 * nov

        # Explanations
        why = []
        why.append(f"Complementarity: their offering matches your goals (score {comp:.2f}).")
        why.append(f"Role fit: {a['type']} ↔ {b['type']} (score {rf:.2f}).")
        shared = sorted(list(set(a_tags).intersection(set(b_tags))))
        if shared:
            why.append("Shared themes: " + ", ".join(shared[:4]) + ".")
        else:
            why.append("Novel connection: low overlap but strong counterparty fit.")

        results.append({
            "match_index": j,
            "score": float(score),
            "why": why,
            "starters": generate_starters(a, b),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
