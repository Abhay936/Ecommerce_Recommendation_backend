"""
Evaluation harness.

WHAT "RELEVANT" MEANS HERE (read this before trusting any number below)
-------------------------------------------------------------------------
This dataset has no user feedback (no clicks, no purchases-by-user, no
ratings-by-user) — see app/data/loader.py's module docstring. That means
we CANNOT compute textbook Precision@K/Recall@K/NDCG against real user
relevance judgments, because there is no ground truth of "what the user
actually wanted."

Instead we use a documented, catalog-derived proxy:
    relevant(query_product) = every other product in the SAME category_id.
This is defensible (same-category is a real, verifiable property of the
data, not fabricated) but it is a PROXY for topical relevance, not for
"the user would have clicked this." A perfect content-based system would
score near-perfectly here almost by construction, which is expected and
is exactly why the comparison against the popularity baseline is the
interesting part of this table, not the absolute numbers.

If real interaction logs are collected later (see README "Future
Improvements"), swap `relevant()` below for real ground truth and every
metric function in metrics.py keeps working unchanged.

Baselines compared
-------------------
  1. Popularity baseline   - same top-N popular products for every query,
                              ignoring the query item entirely.
  2. Content-based only    - pure TF-IDF cosine similarity, no popularity blend.
  3. Popularity-weighted   - stands in for "collaborative filtering" in the
                              comparison table. True CF (user/item-based, ALS,
                              BPR) needs a user-item interaction matrix, which
                              this dataset does not have; see README for what
                              data would be required to add it for real.
  4. Hybrid                - content + popularity + rating (the shipped system).
"""
import logging
import random

import numpy as np
import pandas as pd

from app.config import settings
from app.evaluation import metrics as m
from app.models.artifacts import RecommenderArtifacts
from app.recommendation.hybrid import HybridRecommender

logger = logging.getLogger(__name__)


def _relevant_set(catalog: pd.DataFrame, query_row: int) -> set:
    cat_id = catalog.iloc[query_row]["category_id"]
    same_cat = catalog.index[catalog["category_id"] == cat_id]
    return set(same_cat) - {query_row}


def _popularity_baseline_ranking(popularity_score: pd.Series, exclude_row: int, n: int) -> list:
    ranked = popularity_score.drop(index=exclude_row, errors="ignore").sort_values(ascending=False)
    return list(ranked.head(n).index)


def _content_only_ranking(recommender: HybridRecommender, query_row: int, n: int) -> list:
    sims = recommender._content_scores_for_row(query_row)
    order = np.argsort(-sims)
    order = [int(i) for i in order if i != query_row][:n]
    return order


def _popularity_weighted_ranking(
    recommender: HybridRecommender, query_row: int, n: int, content_weight: float = 0.3
) -> list:
    """
    Stand-in comparison point in place of collaborative filtering (which
    this dataset cannot support — no user-item interactions). Blends a
    LOW weight on content with a HIGH weight on popularity, i.e. "assume
    the query barely matters, mostly recommend generically popular items."
    This is intentionally a weak baseline: it demonstrates why pure
    popularity (regardless of weighting) under-performs true content
    matching on a query-driven, catalog-similarity task.
    """
    sims = recommender._content_scores_for_row(query_row)
    pop = recommender.popularity_score.values
    score = content_weight * sims + (1 - content_weight) * pop
    order = np.argsort(-score)
    order = [int(i) for i in order if i != query_row][:n]
    return order


def _hybrid_ranking(recommender: HybridRecommender, query_row: int, n: int) -> list:
    asin = recommender.catalog.iloc[query_row]["asin"]
    results = recommender.similar_to_product(asin, top_k=n)
    return [recommender.get_row_by_asin(r["asin"]) for r in results]


def run_evaluation(artifacts: RecommenderArtifacts, num_queries: int | None = None, seed: int = 42) -> dict:
    recommender = HybridRecommender(artifacts)
    catalog = recommender.catalog
    n_queries = num_queries or settings.eval_num_queries
    n_queries = min(n_queries, len(catalog))

    rng = random.Random(seed)
    query_rows = rng.sample(range(len(catalog)), n_queries)

    max_k = max(settings.eval_k_values)
    systems = {
        "popularity_baseline": lambda q: _popularity_baseline_ranking(
            recommender.popularity_score, q, max_k
        ),
        "content_based": lambda q: _content_only_ranking(recommender, q, max_k),
        "popularity_weighted_(CF_stand-in)": lambda q: _popularity_weighted_ranking(
            recommender, q, max_k
        ),
        "hybrid": lambda q: _hybrid_ranking(recommender, q, max_k),
    }

    results = {name: {k: [] for k in settings.eval_k_values} for name in systems}
    coverage_lists = {name: [] for name in systems}
    diversity_scores = {name: [] for name in systems}
    map_scores = {name: [] for name in systems}
    mrr_scores = {name: [] for name in systems}

    for q in query_rows:
        relevant = _relevant_set(catalog, q)
        if not relevant:
            continue

        for name, fn in systems.items():
            ranking = fn(q)
            coverage_lists[name].append(ranking)
            map_scores[name].append(m.average_precision_at_k(ranking, relevant, max_k))
            mrr_scores[name].append(m.reciprocal_rank(ranking, relevant))

            cats = catalog.iloc[ranking[:10]]["category_id"].tolist() if ranking else []
            diversity_scores[name].append(m.intra_list_diversity(cats))

            for k in settings.eval_k_values:
                results[name][k].append({
                    "precision": m.precision_at_k(ranking, relevant, k),
                    "recall": m.recall_at_k(ranking, relevant, k),
                    "ndcg": m.ndcg_at_k(ranking, relevant, k),
                    "hit_rate": m.hit_rate_at_k(ranking, relevant, k),
                })

    summary = {}
    for name in systems:
        summary[name] = {
            "MAP": float(np.mean(map_scores[name])) if map_scores[name] else 0.0,
            "MRR": float(np.mean(mrr_scores[name])) if mrr_scores[name] else 0.0,
            "coverage": m.coverage(coverage_lists[name], len(catalog)),
            "diversity@10": float(np.mean(diversity_scores[name])) if diversity_scores[name] else 0.0,
        }
        for k in settings.eval_k_values:
            rows = results[name][k]
            summary[name][f"precision@{k}"] = float(np.mean([r["precision"] for r in rows])) if rows else 0.0
            summary[name][f"recall@{k}"] = float(np.mean([r["recall"] for r in rows])) if rows else 0.0
            summary[name][f"ndcg@{k}"] = float(np.mean([r["ndcg"] for r in rows])) if rows else 0.0
            summary[name][f"hit_rate@{k}"] = float(np.mean([r["hit_rate"] for r in rows])) if rows else 0.0

    return {
        "num_queries_evaluated": len(query_rows),
        "relevance_definition": "same category_id as query product (proxy — see module docstring)",
        "results": summary,
    }


def format_comparison_table(eval_output: dict) -> str:
    results = eval_output["results"]
    k_values = settings.eval_k_values
    headers = ["Model"] + [f"P@{k}" for k in k_values] + [f"R@{k}" for k in k_values] + \
              [f"NDCG@{k}" for k in k_values] + ["MAP", "MRR", "Coverage", "Diversity@10"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for name, vals in results.items():
        row = [name]
        for k in k_values:
            row.append(f"{vals[f'precision@{k}']:.3f}")
        for k in k_values:
            row.append(f"{vals[f'recall@{k}']:.3f}")
        for k in k_values:
            row.append(f"{vals[f'ndcg@{k}']:.3f}")
        row += [f"{vals['MAP']:.3f}", f"{vals['MRR']:.3f}", f"{vals['coverage']:.3f}", f"{vals['diversity@10']:.3f}"]
        lines.append(" | ".join(row))
    return "\n".join(lines)
