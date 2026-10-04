"""
Model persistence.

Everything needed to serve recommendations is fit once (in train.py) and
saved here. The API loads these artifacts at startup instead of
recomputing TF-IDF / popularity on every request (which was the original
bug — build_recommendation_model() was being called inside every route
handler).
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RecommenderArtifacts:
    catalog: pd.DataFrame          # cleaned product catalog, indexed by row_id
    vectorizer: TfidfVectorizer    # fitted TF-IDF vectorizer
    tfidf_matrix: sparse.csr_matrix  # (n_products, n_features) sparse matrix
    popularity_score: pd.Series    # aligned to catalog.index
    metadata: dict                 # training run info: version, timestamp, params, metrics


def _paths(artifacts_dir: str | None = None):
    d = Path(artifacts_dir or settings.artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "catalog": d / "catalog.joblib",
        "vectorizer": d / "tfidf_vectorizer.joblib",
        "matrix": d / "tfidf_matrix.npz",
        "popularity": d / "popularity_score.joblib",
        "metadata": d / "metadata.json",
    }


def save_artifacts(artifacts: RecommenderArtifacts, artifacts_dir: str | None = None) -> None:
    # Catalog / popularity are saved with joblib (pickle-based) rather than
    # parquet to avoid an extra pyarrow dependency for a project this size.
    # For a larger production catalog, parquet + pyarrow would be preferable
    # (columnar, language-agnostic, much smaller on disk).
    paths = _paths(artifacts_dir)

    joblib.dump(artifacts.catalog, paths["catalog"])
    joblib.dump(artifacts.vectorizer, paths["vectorizer"])
    sparse.save_npz(paths["matrix"], artifacts.tfidf_matrix)
    joblib.dump(artifacts.popularity_score, paths["popularity"])

    metadata = dict(artifacts.metadata)
    metadata["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved artifacts to %s", Path(artifacts_dir or settings.artifacts_dir))


def load_artifacts(artifacts_dir: str | None = None) -> RecommenderArtifacts:
    paths = _paths(artifacts_dir)

    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact '{key}' at {p}. Run `python train.py` first."
            )

    catalog = joblib.load(paths["catalog"])
    vectorizer = joblib.load(paths["vectorizer"])
    matrix = sparse.load_npz(paths["matrix"])
    popularity_score = joblib.load(paths["popularity"])
    with open(paths["metadata"]) as f:
        metadata = json.load(f)

    return RecommenderArtifacts(
        catalog=catalog,
        vectorizer=vectorizer,
        tfidf_matrix=matrix,
        popularity_score=popularity_score,
        metadata=metadata,
    )


def artifacts_exist(artifacts_dir: str | None = None) -> bool:
    paths = _paths(artifacts_dir)
    return all(p.exists() for p in paths.values())
