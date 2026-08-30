from __future__ import annotations

import re

import numpy as np
import pandas as pd

RISK_KEYWORDS = {
    "urgency_terms": [
        "urgent",
        "immediate",
        "asap",
        "emergency",
        "quickly",
        "immediately",
        "urgent ah",
        "immediate ah",
    ],
    "safety_terms": [
        "danger",
        "dangerous",
        "unsafe",
        "injury",
        "hurt",
        "slippery",
        "shock",
        "fire",
        "smoke",
        "broken glass",
        "loose railing",
        "safety",
        "risk",
        "hazard",
        "விபத்து",
        "ஆபத்து",
        "பாதுகாப்பு",
    ],
    "electrical_terms": [
        "electrical",
        "electric",
        "current",
        "power outage",
        "switch",
        "wire",
        "panel",
        "shock",
        "மின்சாரம்",
        "மின்பலகை",
    ],
    "health_sanitation_terms": [
        "hygiene",
        "unhygienic",
        "dirty",
        "bad smell",
        "garbage",
        "waste",
        "washroom",
        "toilet",
        "overflow",
        "food quality",
        "water leak",
        "கழிவறை",
        "குப்பை",
        "துர்நாற்றம்",
    ],
    "security_terms": [
        "security",
        "harassment",
        "unauthorized",
        "threat",
        "theft",
        "intruder",
        "gate",
        "guard",
        "பாதுகாப்பு",
    ],
    "outage_terms": [
        "not working",
        "not functioning",
        "unavailable",
        "no water",
        "no internet",
        "power outage",
        "not loading",
        "failed",
        "varala",
        "work aagala",
        "illa",
        "இல்லை",
        "வேலை செய்யவில்லை",
    ],
}


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    normalized_text = text.lower()

    return sum(
        len(re.findall(re.escape(keyword.lower()), normalized_text))
        for keyword in keywords
    )


def extract_text_risk_features(texts: pd.Series | list[str]) -> pd.DataFrame:
    text_series = pd.Series(texts).fillna("").astype(str)

    features: dict[str, list[float]] = {
        "text_length": [],
        "word_count": [],
        "exclamation_count": [],
        "question_count": [],
    }

    for group_name in RISK_KEYWORDS:
        features[f"{group_name}_count"] = []

    for text in text_series:
        cleaned_text = text.strip()

        features["text_length"].append(len(cleaned_text))
        features["word_count"].append(len(cleaned_text.split()))
        features["exclamation_count"].append(cleaned_text.count("!"))
        features["question_count"].append(cleaned_text.count("?"))

        for group_name, keywords in RISK_KEYWORDS.items():
            features[f"{group_name}_count"].append(
                count_keyword_matches(cleaned_text, keywords)
            )

    return pd.DataFrame(features).astype(float)


def get_risk_feature_names() -> list[str]:
    feature_names = [
        "text_length",
        "word_count",
        "exclamation_count",
        "question_count",
    ]

    feature_names.extend(
        f"{group_name}_count"
        for group_name in RISK_KEYWORDS
    )

    return feature_names