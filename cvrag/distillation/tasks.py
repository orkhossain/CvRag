from __future__ import annotations

from ..celery_app import celery_app
from .generator import (
    DISTILLABLE_INTENTS,
    generate_examples,
    output_path,
    save_to_jsonl,
)


@celery_app.task(bind=True, name="distillation.generate")
def generate_distillation_data(
    self,
    intents: list[str] | None = None,
    examples_per_intent: int = 5,
    augment: bool = False,
) -> dict:
    """
    Celery task: generate teacher-labeled training examples and save to JSONL.

    Emits PROGRESS state updates so the caller can poll for live progress.
    Returns a result dict with the output file path and example count.
    """
    task_id = self.request.id
    selected = [i for i in (intents or DISTILLABLE_INTENTS)]
    total_estimate = len(selected) * examples_per_intent

    self.update_state(
        state="PROGRESS",
        meta={"current": 0, "total": total_estimate, "task_id": task_id},
    )

    collected: list[dict] = []

    def _progress(current: int, total: int) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"current": current, "total": total, "task_id": task_id},
        )

    try:
        for example in generate_examples(
            intents=intents,
            examples_per_intent=examples_per_intent,
            augment=augment,
            progress_cb=_progress,
        ):
            collected.append(example)

        out = output_path(task_id)
        save_to_jsonl(collected, out)

        return {
            "task_id": task_id,
            "status": "done",
            "total_examples": len(collected),
            "file": str(out),
        }
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
