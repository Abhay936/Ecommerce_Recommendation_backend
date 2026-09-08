"""
Training pipeline.

    python train.py

Steps:
  1. Load + clean the product catalog (app/data/loader.py).
  2. Fit TF-IDF content features (app/features/text_features.py).
  3. Compute popularity scores (app/features/popularity.py).
  4. Save all artifacts to backend/artifacts/ (app/models/artifacts.py).
  5. Run the evaluation harness and print + save a comparison table.

Re-run this whenever the dataset changes or you tune weights in
app/config.py. The API (app/main.py) loads the saved artifacts at
startup and never retrains on request — that was the main correctness
bug in the original implementation.
"""
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.data.loader import load_products
from app.evaluation.evaluate import format_comparison_table, run_evaluation
from app.features.popularity import compute_popularity_score
from app.features.text_features import fit_tfidf
from app.models.artifacts import RecommenderArtifacts, save_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def main():
    t0 = time.time()

    logger.info("Step 1/5 — loading and cleaning catalog")
    catalog = load_products()
    logger.info("Catalog ready: %d products, %d categories", len(catalog), catalog["category_id"].nunique())

    logger.info("Step 2/5 — fitting TF-IDF content features")
    vectorizer, tfidf_matrix = fit_tfidf(catalog)
    logger.info("TF-IDF matrix: %s, vocabulary size %d", tfidf_matrix.shape, len(vectorizer.vocabulary_))

    logger.info("Step 3/5 — computing popularity scores")
    popularity_score = compute_popularity_score(catalog)

    metadata = {
        "n_products": len(catalog),
        "n_categories": int(catalog["category_id"].nunique()),
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "sample_size_config": settings.sample_size,
        "weights": {
            "content": settings.weight_content,
            "popularity": settings.weight_popularity,
            "rating": settings.weight_rating,
        },
    }

    artifacts = RecommenderArtifacts(
        catalog=catalog,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        popularity_score=popularity_score,
        metadata=metadata,
    )

    logger.info("Step 4/5 — saving artifacts to %s", settings.artifacts_dir)
    save_artifacts(artifacts)

    logger.info("Step 5/5 — running evaluation (%d queries)", settings.eval_num_queries)
    eval_output = run_evaluation(artifacts)
    table = format_comparison_table(eval_output)

    print("\n" + "=" * 100)
    print("MODEL COMPARISON  (relevance proxy: same category_id — see app/evaluation/evaluate.py)")
    print("=" * 100)
    print(table)
    print("=" * 100 + "\n")

    metrics_path = Path(settings.artifacts_dir) / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    logger.info("Saved evaluation results to %s", metrics_path)

    logger.info("Training pipeline finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
