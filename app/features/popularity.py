"""
Popularity scoring.

Used three ways:
  1. Standalone "trending" / "popular" endpoint.
  2. Cold-start fallback for new users (no interaction history) and
     unseen/new products (no content match found).
  3. One term inside the hybrid score, as a quality tie-breaker.

We deliberately do NOT just sort by boughtInLastMonth. A product bought
30,000 times last month with a 2.1-star rating is not "popular" in a
trustworthy sense — it's probably a returns/complaints magnet. We blend
purchase volume, review volume, and rating into one normalized score.

No recency term: the dataset has no timestamp field (no listing date, no
review date), so genuine recency cannot be computed. Adding a recency
component here would mean inventing a fake signal — documented in the
README as a concrete "future improvement" once real timestamped data
exists.
"""
import numpy as np
import pandas as pd

from app.config import settings


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_popularity_score(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series aligned to df.index with a 0-1 popularity score.

    boughtInLastMonth and reviews are heavily right-skewed (a handful of
    viral products dominate), so we log1p-transform before min-max scaling.
    Otherwise the score is basically a boughtInLastMonth indicator and the
    rating term becomes irrelevant.
    """
    bought_norm = _minmax(np.log1p(df["boughtInLastMonth"]))
    reviews_norm = _minmax(np.log1p(df["reviews"]))
    stars_norm = df["stars"].clip(0, 5) / 5.0

    score = (
        settings.pop_weight_bought * bought_norm
        + settings.pop_weight_reviews * reviews_norm
        + settings.pop_weight_stars * stars_norm
    )
    return score.rename("popularity_score")
