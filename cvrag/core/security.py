from fastapi import HTTPException

from .config import API_TOKEN


def guard(token: str | None) -> None:
    if API_TOKEN and token != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
