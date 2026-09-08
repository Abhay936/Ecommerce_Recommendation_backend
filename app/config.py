"""
Central configuration for the recommendation system.

Keeping every tunable value here (instead of scattered magic numbers)
makes the pipeline reproducible and lets train.py / the API share the
exact same settings.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent  # project/


def _env(name: str, default, cast=str):
    """Read RECSYS_<name> from the environment if present, else return default."""
    raw = os.getenv(f"RECSYS_{name.upper()}")
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes")
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    # ---- Data ----
    dataset_path: str = field(
        default_factory=lambda: _env(
            "dataset_path", str(PROJECT_ROOT / "datasets" / "amazon_products.csv")
        )
    )
    # Full dataset is ~1.4M rows. For a laptop-scale demo we sample a
    # stratified subset (see data/loader.py). Set to 0 to use all rows.
    sample_size: int = field(default_factory=lambda: _env("sample_size", 60_000, int))
    random_seed: int = field(default_factory=lambda: _env("random_seed", 42, int))

    # ---- Artifacts ----
    artifacts_dir: str = field(
        default_factory=lambda: _env("artifacts_dir", str(BASE_DIR / "artifacts"))
    )

    # ---- Text features ----
    tfidf_max_features: int = field(
        default_factory=lambda: _env("tfidf_max_features", 20_000, int)
    )
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = field(default_factory=lambda: _env("tfidf_min_df", 2, int))

    # ---- Hybrid scoring weights ----
    # final_score = w_content * content_sim + w_popularity * pop_score + w_rating * rating_score
    # Content similarity dominates because the primary task is
    # "find similar products"; popularity/rating act as quality tie-breakers
    # so we don't recommend a topically-similar but low-quality/unpopular item.
    # See README "Hybrid Weighting" section for the empirical justification.
    weight_content: float = field(default_factory=lambda: _env("weight_content", 0.65, float))
    weight_popularity: float = field(default_factory=lambda: _env("weight_popularity", 0.20, float))
    weight_rating: float = field(default_factory=lambda: _env("weight_rating", 0.15, float))

    # ---- Popularity score blend (used both standalone and inside hybrid) ----
    # popularity_score = pw1 * norm(boughtInLastMonth) + pw2 * norm(reviews) + pw3 * norm(stars)
    pop_weight_bought: float = 0.5
    pop_weight_reviews: float = 0.3
    pop_weight_stars: float = 0.2

    # ---- Serving ----
    default_top_k: int = field(default_factory=lambda: _env("default_top_k", 10, int))
    max_top_k: int = 50
    candidate_pool_size: int = 200  # candidates pulled before final re-ranking

    # ---- Evaluation ----
    eval_k_values: tuple = (5, 10, 20)
    eval_num_queries: int = field(default_factory=lambda: _env("eval_num_queries", 500, int))


settings = Settings()
