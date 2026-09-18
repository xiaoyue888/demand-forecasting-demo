"""Small web-facing helpers shared by the Streamlit application and tests."""

from __future__ import annotations

import os
from urllib.parse import urlsplit


DEFAULT_CASE_STUDY_URL = "http://127.0.0.1:4173/demand-forecasting.html"


def case_study_url() -> str:
    """Return a safe HTTP(S) URL for the portfolio case-study link."""
    value = os.environ.get("DEMO_CASE_STUDY_URL", DEFAULT_CASE_STUDY_URL).strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or "\r" in value
        or "\n" in value
    ):
        return DEFAULT_CASE_STUDY_URL
    return value
