from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
_env_cv_path = os.getenv("CV_PATH")
if _env_cv_path:
    CV_PATH = Path(_env_cv_path)
else:
    _pdf_fallback = BASE_DIR / "cv.pdf"
    CV_PATH = _pdf_fallback if _pdf_fallback.exists() else BASE_DIR / "cv.json"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_TOKEN = os.getenv("API_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "llama-3.3-70b-versatile")
DISTILL_OUTPUT_DIR = BASE_DIR / os.getenv("DISTILL_OUTPUT_DIR", "distillation_output")
DISTILL_EXAMPLES_PER_INTENT = int(os.getenv("DISTILL_EXAMPLES_PER_INTENT", "5"))


def _parse_soft_skills(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


SOFT_SKILLS = _parse_soft_skills(os.getenv("SOFT_SKILLS", ""))
if not SOFT_SKILLS:
    SOFT_SKILLS = [
        "Leadership",
        "Communication",
        "Proactivity",
        "Time Management",
        "Planning",
    ]


def validate_env() -> None:
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. In Hugging Face Spaces, add it under Settings -> Repository secrets."
            )
    elif LLM_PROVIDER in {"hf", "huggingface"}:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set. In Hugging Face Spaces, add it under Settings -> Repository secrets."
            )
    elif LLM_PROVIDER == "ollama":
        # Ollama runs locally; no API key required.
        pass
    else:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use 'groq', 'huggingface', or 'ollama'."
        )
