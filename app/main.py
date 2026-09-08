"""
FastAPI application entrypoint.

    uvicorn app.main:app --reload

Artifacts (TF-IDF vectorizer/matrix, catalog, popularity scores) are
loaded ONCE at startup via the lifespan handler below and attached to
`app.state.recommender`. This replaces the original design where
build_recommendation_model() ran inside every request handler.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.artifacts import artifacts_exist, load_artifacts
from app.recommendation.hybrid import HybridRecommender
from app.routes.product_routes import router as product_router
from app.routes.recommendation_routes import router as recommendation_router
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not artifacts_exist():
        logger.error(
            "No trained artifacts found. Run `python train.py` before starting the API."
        )
        app.state.recommender = None
        app.state.artifacts_metadata = None
    else:
        artifacts = load_artifacts()
        app.state.recommender = HybridRecommender(artifacts)
        app.state.artifacts_metadata = artifacts.metadata
        logger.info(
            "Loaded artifacts: %d products, trained %s",
            len(artifacts.catalog),
            artifacts.metadata.get("saved_at", "unknown"),
        )
    yield
    # No teardown needed — everything is in-memory, process exit cleans up.


app = FastAPI(
    title="E-commerce Recommendation API",
    description="Hybrid (content-based + popularity) product recommendation service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a local demo; restrict in real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(product_router)
app.include_router(recommendation_router)


@app.get("/", tags=["system"])
def root():
    return {"message": "E-commerce Recommendation API is running. See /docs for endpoints."}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    recommender = app.state.recommender
    if recommender is None:
        return HealthResponse(status="degraded", artifacts_loaded=False)
    return HealthResponse(
        status="ok",
        artifacts_loaded=True,
        n_products=len(recommender.catalog),
        model_version=app.state.artifacts_metadata.get("saved_at"),
    )
