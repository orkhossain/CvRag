from __future__ import annotations

from ..celery_app import celery_app
from .generator import DISTILLABLE_INTENTS, generate_examples, output_path, save_to_jsonl


@celery_app.task(bind=True, name="distillation.generate")
def generate_distillation_data(
    self,
    intents: list[str] | None = None,
    examples_per_intent: int = 5,
    augment: bool = False,
) -> dict:
    task_id = self.request.id
    selected = intents or DISTILLABLE_INTENTS

    self.update_state(
        state="PROGRESS",
        meta={"current": 0, "total": len(selected) * examples_per_intent},
    )

    collected: list[dict] = []

    def _progress(current: int, total: int) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"current": current, "total": total},
        )

    try:
        for example in generate_examples(
            intents=intents,
            examples_per_intent=examples_per_intent,
            augment=augment,
            progress_cb=_progress,
        ):
            collected.append(example)

        save_to_jsonl(collected, output_path(task_id))
        return {"task_id": task_id, "total_examples": len(collected)}
    except Exception as exc:
        self.update_state(state="FAILURE", meta={"error": str(exc)})
        raise
