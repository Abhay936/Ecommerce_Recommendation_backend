"""
Text feature extraction for content-based similarity.

TF-IDF is used (rather than dense embeddings) because:
  * Titles are short (a handful of words) and highly keyword-driven
    ("Sion Softside Expandable Roller Luggage") — sparse lexical overlap
    is already a strong, cheap similarity signal for this kind of text.
  * TF-IDF + sparse cosine similarity scales to hundreds of thousands of
    items on a laptop with no GPU and no vector DB.
  * It's fully interpretable: you can point at the shared tokens that
    drove a match, which feeds directly into the explanation strings
    ("similar to products you viewed because of shared terms: ...").

Trade-off / when to switch to embeddings:
  Sentence embeddings (e.g. a small sentence-transformers model) would
  capture semantic similarity beyond shared words (e.g. "sofa" vs
  "couch"), at the cost of needing a model download, GPU/CPU inference
  time per product, and an ANN index (FAISS) once the catalog is large.
  For this dataset's short, keyword-heavy titles the marginal quality
  gain didn't justify that complexity — this is flagged as an "Optional
  Advanced Component" in the README for anyone extending the project.
"""
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings


def build_document(df: pd.DataFrame) -> pd.Series:
    """
    Combine the fields that carry topical meaning into one text blob per
    product. category_id is numeric-only in this dataset (no category name
    lookup table shipped with it), but repeating it as a token still lets
    TF-IDF use "same category" as a similarity signal alongside title text.
    """
    return (
        df["title"].astype(str)
        + " "
        + ("category_" + df["category_id"].astype(str))
        + " "
        + df["isBestSeller"].map({True: "bestseller", False: ""}).astype(str)
    ).str.strip()


def fit_tfidf(df: pd.DataFrame) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit a TF-IDF vectorizer on the training catalog and return (vectorizer, matrix)."""
    documents = build_document(df)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=settings.tfidf_max_features,
        ngram_range=settings.tfidf_ngram_range,
        min_df=settings.tfidf_min_df,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def transform_new(vectorizer: TfidfVectorizer, df: pd.DataFrame) -> sparse.csr_matrix:
    """
    Vectorize products NOT seen during fit_tfidf (new-product cold start).
    Safe to call with a single-row DataFrame.
    """
    documents = build_document(df)
    return vectorizer.transform(documents)
