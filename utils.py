"""Utility functions for data processing and display formatting."""

import re
from collections import Counter


def clean_genres(x):
    """Clean genre data from various formats (list, Counter, string) into a list of genre strings."""
    if isinstance(x, (list, tuple)):
        return list(x)
    elif isinstance(x, Counter):
        return list(x.keys())
    elif isinstance(x, str):
        return [g.strip().lower() for g in x.strip("{}").replace('"', "").split(",") if g.strip()]
    else:
        return []


def format_label(s):
    """
    Format string for display with typical English titling standards.
    Converts 'alternative_rock' or 'alternativeRock' to 'Alternative Rock'.
    """
    if not s or not isinstance(s, str):
        return str(s)
    cleaned = s.replace("_", " ")
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    return cleaned.strip().title()
