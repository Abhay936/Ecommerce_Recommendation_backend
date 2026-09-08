import pandas as pd

from app.data.loader import clean, stratified_sample


def _raw_df():
    return pd.DataFrame({
        "asin": ["A1", "A1", "A2", "A3", "A4"],  # A1 duplicated
        "title": ["Blue Widget", None, "", "Red Gadget", "Green Gizmo"],
        "imgUrl": ["u1", "u1", "u2", "u3", "u4"],
        "productURL": ["p1", "p1", "p2", "p3", "p4"],
        "stars": [4.5, 4.5, None, 6.2, -1],  # out-of-range values
        "reviews": [10, 10, -5, None, 100],
        "price": [9.99, 9.99, None, 19.99, -5],
        "listPrice": [None, None, 0, 0, 25.0],
        "category_id": [1, 1, 2, None, 1],
        "isBestSeller": [True, True, False, None, True],
        "boughtInLastMonth": [100, 100, None, 50, -10],
    })


def test_clean_drops_duplicate_asins():
    df = clean(_raw_df())
    assert df["asin"].duplicated().sum() == 0


def test_clean_drops_empty_titles():
    df = clean(_raw_df())
    assert (df["title"] == "").sum() == 0


def test_clean_clips_out_of_range_stars():
    df = clean(_raw_df())
    assert df["stars"].between(0, 5).all()


def test_clean_fills_missing_numeric_with_zero_and_clips_negatives():
    df = clean(_raw_df())
    assert (df["reviews"] >= 0).all()
    assert (df["boughtInLastMonth"] >= 0).all()
    assert (df["price"] >= 0).all()


def test_clean_assigns_stable_row_id():
    df = clean(_raw_df())
    assert list(df["row_id"]) == list(range(len(df)))


def test_stratified_sample_keeps_rare_categories():
    df = pd.DataFrame({
        "asin": [f"A{i}" for i in range(100)],
        "title": [f"Product {i}" for i in range(100)],
        "category_id": [1] * 95 + [2] * 5,  # category 2 is rare
    })
    sampled = stratified_sample(df, n=20, seed=0)
    assert (sampled["category_id"] == 2).sum() >= 1


def test_stratified_sample_returns_full_df_when_n_exceeds_size():
    df = pd.DataFrame({"asin": ["A1", "A2"], "category_id": [1, 2]})
    sampled = stratified_sample(df, n=1000, seed=0)
    assert len(sampled) == len(df)
