from typing import Dict, Optional

from pydantic import BaseModel


class Q(BaseModel):
    query: str
    session_id: Optional[str] = None


class CVPayload(BaseModel):
    cv: Dict
