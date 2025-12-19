from typing import Dict

from pydantic import BaseModel


class Q(BaseModel):
    query: str


class CVPayload(BaseModel):
    cv: Dict
