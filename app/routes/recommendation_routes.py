"""
Recommendation endpoints.

All routes read from `request.app.state.recommender`, a HybridRecommender
built once at startup from saved artifacts (see app/main.py). No route
handler ever refits TF-IDF or touches the raw CSV/DB — that was the
original bug (model rebuilt on every request).
"""
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.config import settings
from app.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter(tags=["recommendations"])
logger = logging.getLogger(__name__)


def _clamp_top_k(top_k: int) -> int:
    return max(1, min(top_k, settings.max_top_k))


@router.get("/recommendations/{asin}", response_model=RecommendationResponse)
def get_recommendations(request: Request, asin: str, top_k: int = Query(default=settings.default_top_k)):
    recommender = request.app.state.recommender
    top_k = _clamp_top_k(top_k)

    if not recommender.product_exists(asin):
        # New-product cold start: we don't know this asin at all. Rather than
        # 404 with nothing useful, fall back to trending so the endpoint is
        # still helpful — but the reason string makes the fallback explicit.
        recs = recommender.trending(top_k=top_k, reason="new_product_no_text_match")
        return {"asin": asin, "recommendations": recs, "strategy": "cold_start_trending"}

    recs = recommender.similar_to_product(asin, top_k=top_k)
    return {"asin": asin, "recommendations": recs, "strategy": "hybrid_item_to_item"}


@router.get("/similar-products/{asin}", response_model=RecommendationResponse)
def get_similar_products(request: Request, asin: str, top_k: int = Query(default=settings.default_top_k)):
    """Alias kept distinct from /recommendations for API clarity (per spec)."""
    return get_recommendations(request, asin, top_k)


@router.get("/trending", response_model=RecommendationResponse)
def get_trending(
    request: Request,
    top_k: int = Query(default=settings.default_top_k),
    category_id: int | None = Query(default=None),
):
    recommender = request.app.state.recommender
    top_k = _clamp_top_k(top_k)
    recs = recommender.trending(top_k=top_k, category_id=category_id)
    return {"asin": None, "recommendations": recs, "strategy": "popularity"}


@router.post("/recommendations", response_model=RecommendationResponse)
def post_recommendations(request: Request, body: RecommendationRequest):
    recommender = request.app.state.recommender
    top_k = _clamp_top_k(body.top_k)

    unknown = [a for a in body.viewed_asins if not recommender.product_exists(a)]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown asin(s), not present in catalog: {unknown}",
        )

    recs = recommender.recommend_for_user(body.viewed_asins, top_k=top_k)
    strategy = "session_content_based" if body.viewed_asins else "cold_start_trending"
    return {"asin": None, "recommendations": recs, "strategy": strategy}


@router.get("/search", response_model=RecommendationResponse)
def search(request: Request, q: str = Query(..., min_length=1), top_k: int = Query(default=settings.default_top_k)):
    recommender = request.app.state.recommender
    top_k = _clamp_top_k(top_k)
    recs = recommender.search(q, top_k=top_k)
    return {"asin": None, "recommendations": recs, "strategy": "content_search"}
