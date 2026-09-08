import pandas as pd
import pytest

from app.features.popularity import compute_popularity_score
from app.features.text_features import fit_tfidf
from app.models.artifacts import RecommenderArtifacts
from app.recommendation.hybrid import HybridRecommender


def _catalog():
    return pd.DataFrame({
        "asin": [f"A{i}" for i in range(6)],
        "title": [
            "Blue Running Shoes",
            "Red Running Shoes",
            "White Running Sneakers",
            "Kitchen Blender",
            "Electric Kettle",
            "Stainless Steel Blender",
        ],
        "imgUrl": ["img"] * 6,
        "price": [50.0, 45.0, 60.0, 30.0, 20.0, 35.0],
        "stars": [4.8, 3.0, 4.5, 4.0, 3.5, 4.2],
        "category_id": [10, 10, 10, 20, 20, 20],
        "isBestSeller": [True, False, False, False, False, True],
        "reviews": [1000, 5, 50, 20, 10, 200],
        "boughtInLastMonth": [500, 0, 10, 5, 0, 40],
        "row_id": list(range(6)),
    })


@pytest.fixture
def recommender():
    df = _catalog()
    vectorizer, matrix = fit_tfidf(df)
    pop = compute_popularity_score(df)
    artifacts = RecommenderArtifacts(
        catalog=df, vectorizer=vectorizer, tfidf_matrix=matrix,
        popularity_score=pop, metadata={},
    )
    return HybridRecommender(artifacts)


def test_similar_to_product_excludes_itself(recommender):
    results = recommender.similar_to_product("A0", top_k=5)
    assert all(r["asin"] != "A0" for r in results)


def test_similar_to_product_prefers_same_category(recommender):
    results = recommender.similar_to_product("A0", top_k=2)
    returned_asins = {r["asin"] for r in results}
    # A1 and A2 are the same-category shoe products; a blender should not
    # outrank them for a shoe query.
    assert returned_asins.issubset({"A1", "A2"})


def test_unknown_product_returns_none_for_caller_to_handle_cold_start(recommender):
    assert recommender.similar_to_product("DOES_NOT_EXIST", top_k=5) is None


def test_trending_returns_top_k_by_popularity(recommender):
    results = recommender.trending(top_k=3)
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_trending_scoped_to_category(recommender):
    results = recommender.trending(top_k=10, category_id=10)
    assert all(r["category_id"] == 10 for r in results)


def test_trending_unknown_category_falls_back_to_global(recommender):
    results = recommender.trending(top_k=3, category_id=9999)
    assert len(results) == 3  # doesn't silently return zero results


def test_new_user_no_history_falls_back_to_trending(recommender):
    results = recommender.recommend_for_user([], top_k=3)
    assert len(results) == 3
    assert results[0]["reason"] == "Popular right now (no browsing history yet)"


def test_session_based_recommendation_excludes_viewed_items(recommender):
    results = recommender.recommend_for_user(["A0"], top_k=5)
    assert all(r["asin"] != "A0" for r in results)


def test_new_product_cold_start_matches_similar_text(recommender):
    results = recommender.similar_to_new_product("Running Shoes for Trail", category_id=10, top_k=2)
    returned_asins = {r["asin"] for r in results}
    assert returned_asins & {"A0", "A1", "A2"}  # should find at least one running-shoe product


def test_new_product_cold_start_no_text_match_falls_back(recommender):
    results = recommender.similar_to_new_product("Xyzzyplonk Frobnicator 9000", category_id=10, top_k=2)
    assert all(r["reason"] == "Popular in this category (no close text match found)" for r in results)


def test_search_returns_relevant_results(recommender):
    results = recommender.search("running shoes", top_k=3)
    returned_asins = {r["asin"] for r in results}
    assert returned_asins & {"A0", "A1", "A2"}


def test_every_recommendation_has_a_reason_string(recommender):
    for results in [
        recommender.similar_to_product("A0", top_k=3),
        recommender.trending(top_k=3),
        recommender.search("blender", top_k=3),
    ]:
        assert all(isinstance(r["reason"], str) and r["reason"] for r in results)
