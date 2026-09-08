"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    asin: str
    title: str
    imgUrl: str
    price: float
    stars: float
    category_id: int
    score: float
    content_similarity: float | None = None
    reason: str


class RecommendationResponse(BaseModel):
    asin: str | None = None
    recommendations: list[RecommendationItem]
    strategy: str


class RecommendationRequest(BaseModel):
    """Body for POST /recommendations — session-based, no persisted user table."""
    viewed_asins: list[str] = Field(default_factory=list, description="ASINs viewed this session")
    top_k: int = Field(default=10, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    n_products: int | None = None
    model_version: str | None = None
