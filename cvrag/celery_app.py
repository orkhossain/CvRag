from celery import Celery

from .core.config import REDIS_URL

celery_app = Celery(
    "cvrag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["cvrag.distillation.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=86400,  # 24 hours
    worker_prefetch_multiplier=1,
)
