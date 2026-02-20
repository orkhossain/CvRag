from __future__ import annotations

from typing import Iterable, Optional

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

_LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "cs": "Czech",
}


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "auto"
    try:
        code = detect(text)
    except Exception:
        return "auto"
    return _LANGUAGE_NAMES.get(code, code)


def resolve_language(preferred: Optional[str], samples: Iterable[str]) -> str:
    if preferred:
        preferred = preferred.strip()
        if preferred:
            return preferred
    merged = " ".join(sample for sample in samples if sample)
    detected = detect_language(merged)
    return "English" if detected == "auto" else detected
