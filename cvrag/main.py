from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import ALLOWED_ORIGINS, validate_env
from .core.cv_data import ensure_cv_json
from .rag.retriever import init_retriever

validate_env()
ensure_cv_json()
init_retriever()

app = FastAPI(title="CV Ask API (HF Spaces)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
