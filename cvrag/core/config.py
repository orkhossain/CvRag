from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CV_PATH = Path(os.getenv("CV_PATH", BASE_DIR / "cv.json"))

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_TOKEN = os.getenv("API_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


def validate_env() -> None:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. In Hugging Face Spaces, add it under Settings -> Repository secrets."
        )
