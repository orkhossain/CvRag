from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core.security import guard
from ..distillation.generator import DISTILLABLE_INTENTS, output_path
from ..distillation.tasks import generate_distillation_data

router = APIRouter(prefix="/distill", tags=["distillation"])


class DistillStartRequest(BaseModel):
    intents: list[str] | None = Field(
        default=None,
        description=f"Subset of intents to generate. Defaults to all: {DISTILLABLE_INTENTS}",
    )
    examples_per_intent: int = Field(default=5, ge=1, le=50)
    augment_queries: bool = Field(
        default=False,
        description="Use teacher LLM to generate extra query paraphrases when seeds run out",
    )


class DistillStartResponse(BaseModel):
    task_id: str
    status: str
    intents: list[str]
    examples_per_intent: int


class DistillStatusResponse(BaseModel):
    task_id: str
    status: str
    current: int | None = None
    total: int | None = None
    total_examples: int | None = None
    error: str | None = None


@router.post("/start", response_model=DistillStartResponse)
def start_distillation(
    payload: DistillStartRequest,
    authorization: str | None = Header(default=None),
):
    guard(authorization)

    intents = payload.intents or DISTILLABLE_INTENTS
    invalid = [i for i in intents if i not in DISTILLABLE_INTENTS]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown intents: {invalid}. Valid: {DISTILLABLE_INTENTS}",
        )

    task = generate_distillation_data.delay(
        intents=intents,
        examples_per_intent=payload.examples_per_intent,
        augment=payload.augment_queries,
    )

    return DistillStartResponse(
        task_id=task.id,
        status="queued",
        intents=intents,
        examples_per_intent=payload.examples_per_intent,
    )


@router.get("/status/{task_id}", response_model=DistillStatusResponse)
def distillation_status(
    task_id: str,
    authorization: str | None = Header(default=None),
):
    guard(authorization)

    from celery.result import AsyncResult
    from ..celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)
    state = result.state

    if state == "PENDING":
        return DistillStatusResponse(task_id=task_id, status="pending")

    if state == "PROGRESS":
        meta = result.info or {}
        return DistillStatusResponse(
            task_id=task_id,
            status="running",
            current=meta.get("current"),
            total=meta.get("total"),
        )

    if state == "SUCCESS":
        info = result.result or {}
        return DistillStatusResponse(
            task_id=task_id,
            status="done",
            total_examples=info.get("total_examples"),
        )

    if state == "FAILURE":
        return DistillStatusResponse(
            task_id=task_id,
            status="failed",
            error=str(result.info),
        )

    return DistillStatusResponse(task_id=task_id, status=state.lower())


@router.get("/download/{task_id}")
def download_distillation(
    task_id: str,
    authorization: str | None = Header(default=None),
):
    guard(authorization)

    path = output_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready or task_id unknown")

    return FileResponse(
        path=path,
        filename=f"distillation_{task_id}.jsonl",
        media_type="application/x-ndjson",
    )


@router.get("/intents")
def list_intents(authorization: str | None = Header(default=None)):
    guard(authorization)
    return {"intents": DISTILLABLE_INTENTS}
