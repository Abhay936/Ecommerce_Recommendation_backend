"""
API integration tests.

Requires trained artifacts to exist (run `python train.py` first) since
these hit the real lifespan startup, exactly like the production app.
We use the small in-memory fixture catalog by pointing RECSYS_ARTIFACTS_DIR
at a temp directory built with a tiny synthetic dataset, so tests don't
depend on the full 1.4M-row CSV being present or the 60k sample being
pre-trained.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.features.popularity import compute_popularity_score
from app.features.text_features import fit_tfidf
from app.models.artifacts import RecommenderArtifacts, save_artifacts


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    artifacts_dir = tmp_path_factory.mktemp("artifacts")

    df = pd.DataFrame({
        "asin": [f"A{i}" for i in range(6)],
        "title": [
            "Blue Running Shoes", "Red Running Shoes", "White Running Sneakers",
            "Kitchen Blender", "Electric Kettle", "Stainless Steel Blender",
        ],
        "imgUrl": ["img"] * 6,
        "productURL": ["url"] * 6,
        "price": [50.0, 45.0, 60.0, 30.0, 20.0, 35.0],
        "listPrice": [0.0] * 6,
        "stars": [4.8, 3.0, 4.5, 4.0, 3.5, 4.2],
        "category_id": [10, 10, 10, 20, 20, 20],
        "isBestSeller": [True, False, False, False, False, True],
        "reviews": [1000, 5, 50, 20, 10, 200],
        "boughtInLastMonth": [500, 0, 10, 5, 0, 40],
        "row_id": list(range(6)),
    })
    vectorizer, matrix = fit_tfidf(df)
    pop = compute_popularity_score(df)
    artifacts = RecommenderArtifacts(
        catalog=df, vectorizer=vectorizer, tfidf_matrix=matrix,
        popularity_score=pop, metadata={"n_products": 6},
    )
    save_artifacts(artifacts, artifacts_dir=str(artifacts_dir))

    # Point settings at the temp artifacts dir before importing the app so
    # the lifespan startup loads OUR fixture, not whatever is in backend/artifacts.
    object.__setattr__(settings, "artifacts_dir", str(artifacts_dir))

    from app.main import app  # imported after settings patch
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_products"] == 6


def test_get_recommendations_for_known_product(client):
    resp = client.get("/recommendations/A0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "hybrid_item_to_item"
    assert all(r["asin"] != "A0" for r in body["recommendations"])


def test_get_recommendations_for_unknown_product_falls_back(client):
    resp = client.get("/recommendations/DOES_NOT_EXIST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "cold_start_trending"
    assert len(body["recommendations"]) > 0


def test_trending_endpoint(client):
    resp = client.get("/trending?top_k=3")
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) == 3


def test_trending_scoped_to_category(client):
    resp = client.get("/trending?top_k=10&category_id=10")
    body = resp.json()
    assert all(r["category_id"] == 10 for r in body["recommendations"])


def test_search_endpoint(client):
    resp = client.get("/search?q=running shoes&top_k=3")
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) > 0


def test_search_requires_query_param(client):
    resp = client.get("/search")
    assert resp.status_code == 422  # missing required `q`


def test_post_recommendations_with_viewed_items(client):
    resp = client.post("/recommendations", json={"viewed_asins": ["A0"], "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert all(r["asin"] != "A0" for r in body["recommendations"])


def test_post_recommendations_with_unknown_asin_returns_422(client):
    resp = client.post("/recommendations", json={"viewed_asins": ["NOPE"], "top_k": 3})
    assert resp.status_code == 422


def test_post_recommendations_empty_history_is_cold_start(client):
    resp = client.post("/recommendations", json={"viewed_asins": [], "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "cold_start_trending"


def test_get_products_paginated(client):
    resp = client.get("/products?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 6
    assert len(body["products"]) == 2


def test_get_single_product(client):
    resp = client.get("/products/A0")
    assert resp.status_code == 200
    assert resp.json()["asin"] == "A0"


def test_get_single_product_not_found(client):
    resp = client.get("/products/NOPE")
    assert resp.status_code == 404


def test_top_k_is_clamped_to_max(client):
    resp = client.get(f"/trending?top_k={settings.max_top_k + 100}")
    assert len(resp.json()["recommendations"]) <= settings.max_top_k
