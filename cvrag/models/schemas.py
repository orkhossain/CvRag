from typing import Dict, List, Optional

from pydantic import BaseModel


class Q(BaseModel):
    query: str
    session_id: Optional[str] = None
    language: Optional[str] = None


class CVPayload(BaseModel):
    cv: Dict


class RoleFitRequest(BaseModel):
    job_description: str
    role: Optional[str] = None
    company: Optional[str] = None
    language: Optional[str] = None


class QuickSummaryRequest(BaseModel):
    role_level: Optional[str] = None
    focus: Optional[str] = None
    language: Optional[str] = None


class ProjectDeepDiveRequest(BaseModel):
    project_name: Optional[str] = None
    limit: Optional[int] = 3
    focus: Optional[str] = None
    language: Optional[str] = None


class StarBankRequest(BaseModel):
    competencies: Optional[List[str]] = None
    count: Optional[int] = 3
    language: Optional[str] = None


class ExportRequest(BaseModel):
    format: str
    focus: Optional[str] = None
    language: Optional[str] = None
