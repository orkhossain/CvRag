import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.distillation_routes import router as distillation_router
from .api.routes import router
from .celery_app import celery_app
from .core.config import ALLOWED_ORIGINS, validate_env
from .core.cv_data import ensure_cv_json
from .rag.retriever import init_retriever

validate_env()
ensure_cv_json()
init_retriever()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Embedded Celery worker — runs in a daemon thread so no external process is needed.
    # Uses memory:// broker, so broker and worker share the same Python process.
    worker = celery_app.Worker(pool="threads", concurrency=2, loglevel="warning")
    t = threading.Thread(target=worker.start, daemon=True)
    t.start()
    yield
    worker.stop()


app = FastAPI(title="CV Ask API (HF Spaces)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(distillation_router)
