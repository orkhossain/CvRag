from celery import Celery

# memory:// broker + cache+memory:// backend — no Redis or external process needed.
# The worker runs embedded in the FastAPI process (see cvrag/main.py lifespan).
celery_app = Celery(
    "cvrag",
    broker="memory://",
    backend="cache+memory://",
    include=["cvrag.distillation.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=86400,
    worker_pool="threads",
    worker_concurrency=2,
)
