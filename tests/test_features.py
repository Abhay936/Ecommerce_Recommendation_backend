import pandas as pd

from app.features.popularity import compute_popularity_score
from app.features.text_features import build_document, fit_tfidf, transform_new


def _catalog():
    return pd.DataFrame({
        "asin": ["A1", "A2", "A3"],
        "title": ["Blue Running Shoes", "Red Running Shoes", "Kitchen Blender"],
        "category_id": [10, 10, 20],
        "isBestSeller": [True, False, False],
        "stars": [4.8, 3.0, 4.0],
        "reviews": [1000, 5, 50],
        "boughtInLastMonth": [500, 0, 10],
    })


def test_popularity_score_in_range_0_1():
    df = _catalog()
    scores = compute_popularity_score(df)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_popularity_score_ranks_bestseller_highest():
    df = _catalog()
    scores = compute_popularity_score(df)
    assert scores.iloc[0] > scores.iloc[1]
    assert scores.iloc[0] > scores.iloc[2]


def test_popularity_score_constant_column_does_not_crash():
    df = _catalog()
    df["boughtInLastMonth"] = 0  # all-equal column (edge case for min-max scaling)
    scores = compute_popularity_score(df)
    assert not scores.isna().any()


def test_build_document_includes_category_and_bestseller_tokens():
    df = _catalog()
    docs = build_document(df)
    assert "category_10" in docs.iloc[0]
    assert "bestseller" in docs.iloc[0]
    assert "bestseller" not in docs.iloc[1]


def test_fit_tfidf_shapes_match_catalog():
    df = _catalog()
    vectorizer, matrix = fit_tfidf(df)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(vectorizer.vocabulary_)


def test_transform_new_product_uses_fitted_vocabulary():
    df = _catalog()
    vectorizer, _ = fit_tfidf(df)
    new_product = pd.DataFrame([{
        "title": "Blue Running Shoes for Men",
        "category_id": 10,
        "isBestSeller": False,
    }])
    vec = transform_new(vectorizer, new_product)
    assert vec.shape[0] == 1
    assert vec.nnz > 0  # should share vocabulary with existing "running shoes" products
