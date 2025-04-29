import ast
import re
from collections import Counter
from typing import List, Dict, Optional, Set

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from difflib import SequenceMatcher

# Load spaCy model for noun chunk extraction, normalisation and embeddings
nlp = spacy.load("en_core_web_md")


# Generic exclusion phrases
_EXCLUDE_PHRASES = {
    "experience",
    "skills",
    "data",
    "ability",
    "knowledge",
    "team",
    "work",
    "requirements",
    "years",
    "use",
    "using",
    "related field",
    "familiarity",
    "strong foundation",
    "a related field",
    "monolix",
    "simbiology",
    "phdmasters",
    "vllm",
}


def load_skill_ontologies() -> Dict[str, List[str]]:
    """
    Return a mapping from macro-area names to lists of curated skill keywords.

    The ontologies cover key domains in data science and engineering:
    - Data Science practices
    - ML techniques
    - Statistical techniques
    - ML frameworks
    - DevOps
    - MLOps
    - Cloud platforms and services
    - Programming languages
    """
    ontologies: Dict[str, List[str]] = {
        "data_science_practices": [
            "data cleansing",
            "data preprocessing",
            "data wrangling",
            "feature engineering",
            "exploratory analysis",
            "visualization",
            "evaluation",
            "training",
            "monitoring",
            "cross-validation",
            "a/b testing",
            "pipeline development",
        ],
        "ml_techniques": [
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
            "classification",
            "regression",
            "clustering",
            "dimensionality reduction",
            "anomaly detection",
            "recommendation systems",
            "time series forecasting",
            "natural language processing",
            "nlp",
            "cnn",
            "rnn",
            "lstm",
            "computer vision",
            "deep learning",
            "transfer learning",
            "generative models",
            "ensemble methods",
            "active learning",
            "semi-supervised learning",
            "multi-task learning",
            "meta-learning",
            "federated learning",
            "graph neural networks",
            "self-supervised learning",
            "contrastive learning",
            "attention mechanisms",
            "transformer",
            "diffusion",
        ],
        "statistical_techniques": [
            "hypothesis testing",
            "probability distributions",
            "statistical inference",
            "bayesian",
            "experimental design",
            "multivariate",
            "confidence intervals",
            "correlation",
            "survival",
            "time series",
            "signal processing",
            "causal inference",
        ],
        "ml_frameworks": [
            "tensorflow",
            "pytorch",
            "scikit learn",
            "keras",
            "xgboost",
            "lightgbm",
            "catboost",
            "fastai",
            "h2o",
            "mlpack",
            "shogun",
        ],
        "devops": [
            "docker",
            "kubernetes",
            "ci/cd",
            "terraform",
            "ansible",
            "jenkins",
            "circleci",
            "github actions",
            "helm",
            "prometheus",
            "grafana",
            "vault",
        ],
        "mlops": [
            "mlflow",
            "kubeflow",
            "airflow",
            "seldon",
            "tensorboard",
            "pachyderm",
            "dvc",
            "neptune",
            "clearml",
            "dagster",
            "feast",
            "bentoml",
        ],
        "cloud": [
            "aws",
            "azure",
            "gcp",
            "s3",
            "ec2",
            "lambda",
            "bigquery",
            "databricks",
            "redshift",
            "emr",
            "cloudformation",
            "azure ml",
        ],
        "programming_languages": [
            "python",
            "r language",
            "java",
            "scala",
            "sql",
            "c++ language",
            "c# language",
            "go language",
            "javascript",
            "bash",
            "ruby",
            "matlab",
        ],
    }
    return ontologies


def embed_phrases(phrases: List[str]) -> np.ndarray:
    return np.vstack([nlp(p).vector for p in phrases])


def _normalise(term: str) -> str:
    """
    Lowercase, strip out punctuation, collapse whitespace.
    """
    t = term.strip().lower()
    t = re.sub(r"[^a-z0-9\s]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def filter_by_ontologies(
    cleaned_phrases: List[List[str]],
    ontologies: Dict[str, List[str]],
    fuzzy_threshold: Optional[float] = None,
) -> List[List[str]]:
    """
    Keep only those phrases whose normalized form appears in your ontologies (exactly),
    or (if fuzzy_threshold is set) whose embedding similarity to any ontology term
    exceeds that threshold.
    """
    # Build normalized-truth set
    canonical: Set[str] = {
        _normalise(term) for terms in ontologies.values() for term in terms
    }

    # Precompute embeddings if we need fuzzy matching
    ont_terms = list(canonical)
    if fuzzy_threshold is not None:
        ont_embeds = np.vstack([nlp(t).vector for t in ont_terms])

    out: List[List[str]] = []
    for phrases in cleaned_phrases:
        keep: List[str] = []
        for p in phrases:
            norm = _normalise(p)
            if norm in canonical:
                keep.append(p)
            elif fuzzy_threshold is not None:
                # compute similarity to all ontology terms
                vec = nlp(norm).vector
                sims = (ont_embeds @ vec) / (
                    np.linalg.norm(ont_embeds, axis=1) * np.linalg.norm(vec) + 1e-8
                )
                if sims.max() >= fuzzy_threshold:
                    keep.append(p)
        out.append(keep)
    return out


def replace_keywords_with_language(data: List[List[str]]) -> List[List[str]]:
    """
    Replaces isolated occurrences of 'r', 'c#', and 'c++' with their respective
    'language' versions inside a nested list of strings, ensuring they are treated
    correctly downstream.
    """
    pattern = r"(?<=\b)(r|go|c\+\+|c#)(?=\b)"
    updated_data = []

    for inner_list in data:
        updated_inner = []
        for item in inner_list:
            new_item = re.sub(
                pattern,
                lambda match: f"{match.group(0)} language",
                item,
                flags=re.IGNORECASE,
            )
            updated_inner.append(new_item)
        updated_data.append(updated_inner)

    return updated_data


def load_and_parse_requirements(
    filepath: str, requirements_col: str = "requirements"
) -> pd.DataFrame:
    """Read CSV and parse a JSON-like requirements column into flat lowercase text."""
    df = pd.read_csv(filepath)

    def _parse(cell: str) -> str:
        """
        Parse a single cell from the requirements column.
        """
        # If the cell is NaN, return an empty string
        if pd.isna(cell):
            return ""
        # Strip leading/trailing whitespace
        cell = cell.strip()
        # Check if the cell contains a JSON-like list (starts with "[" and ends with "]")
        if cell.startswith("[") and cell.endswith("]"):
            try:
                # Evaluate the string as a Python literal (e.g., a list)
                items = ast.literal_eval(cell)
                # If the evaluated value is a list, join its items into a single string
                if isinstance(items, list):
                    return " ".join(str(i) for i in items)
            except Exception:
                pass
        return cell

    df["req_text"] = df[requirements_col].apply(_parse).str.lower()
    return df


def extract_noun_phrases(texts: List[str]) -> List[List[str]]:
    """Extract noun chunks from texts using spaCy."""
    all_phrases = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        phrases = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        all_phrases.append(phrases)
    return all_phrases


def filter_phrases_by_df(
    doc_phrases: List[List[str]], min_df: float = 0.01, max_df: float = 0.8
) -> List[List[str]]:
    """Filter phrases based on document frequency thresholds."""
    total_docs = len(doc_phrases)
    df_counts = Counter()
    for phrases in doc_phrases:
        for p in set(phrases):
            df_counts[p] += 1
    lower_thresh = min_df * total_docs
    upper_thresh = max_df * total_docs
    filtered = []
    for phrases in doc_phrases:
        filtered.append(
            [p for p in phrases if lower_thresh <= df_counts[p] <= upper_thresh]
        )
    return filtered


def clean_phrases(doc_phrases: List[List[str]]) -> List[List[str]]:
    """Filter out irrelevant or stop-word phrases by POS and token rules."""
    cleaned = []
    for phrases in doc_phrases:
        good = []
        for phrase in phrases:
            p = re.sub(r"\b(e\.g\.|i\.e\.)\b", "", phrase)
            p = re.sub(r"[^a-z0-9\s]+", "", p.lower()).strip()
            if len(p) < 3 or p in _EXCLUDE_PHRASES:
                continue
            doc = nlp(p)
            if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc):
                continue
            content = [
                tok for tok in doc if tok.is_alpha and tok.lower_ not in STOP_WORDS
            ]
            if not content or any(tok.like_num for tok in doc):
                continue
            good.append(p)
        cleaned.append(good)
    return cleaned


def filter_phrases_by_idf(
    cleaned_phrases: List[List[str]], threshold_percentile: float = 25.0
) -> List[List[str]]:
    """Remove extremely common phrases using IDF: drop phrases with low idf."""
    docs = [" ".join(ph) for ph in cleaned_phrases]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z0-9][a-z0-9\s]+\b", lowercase=True, norm=None
    )
    vectorizer.fit(docs)
    idf = vectorizer.idf_
    cutoff = np.percentile(idf, threshold_percentile)
    vocab = vectorizer.vocabulary_
    filtered = []
    for phrases in cleaned_phrases:
        filtered.append(
            [p for p in phrases if vocab.get(p) is not None and idf[vocab[p]] >= cutoff]
        )
    return filtered


def map_phrases_to_clusters(
    cleaned_phrases: List[List[str]],
    ontologies: Dict[str, List[str]],
    fuzzy_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Assign each unique cleaned phrase to an ontology cluster, with fallbacks:
      1) Exact match on normalised form.
      2) Substring match (e.g. 'correlation' in 'correlation analysis').
      3) Fuzzy match by string similarity (SequenceMatcher) above threshold.

    Returns a DataFrame with columns: term, freq, cluster.
    """

    # Build reverse lookup: normalised ontology term -> cluster
    term_to_cluster: Dict[str, str] = {}
    for cluster, terms in ontologies.items():
        for t in terms:
            term_to_cluster[_normalise(t)] = cluster

    # Count phrase frequencies
    phrase_counts = Counter(p for phrases in cleaned_phrases for p in phrases)

    records = []
    for term, freq in phrase_counts.items():
        norm = _normalise(term)
        cluster = term_to_cluster.get(norm)

        # 2) Substring fallback
        if cluster is None:
            for t_norm, cl in term_to_cluster.items():
                if norm in t_norm or t_norm in norm:
                    cluster = cl
                    break

        # 3) Fuzzy fallback
        if cluster is None and fuzzy_threshold is not None:
            best_ratio = 0.0
            best_cluster = None
            for t_norm, cl in term_to_cluster.items():
                ratio = SequenceMatcher(None, norm, t_norm).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cluster = cl
            if best_ratio >= fuzzy_threshold:
                cluster = best_cluster

        if cluster is None:
            cluster = "Other"

        records.append({"term": term, "freq": freq, "cluster": cluster})

    return pd.DataFrame.from_records(records, columns=["term", "freq", "cluster"])


def semantic_cluster_phrases(
    skill_df: pd.DataFrame, n_clusters: int = 6, linkage: str = "average"
) -> pd.DataFrame:
    """Cluster unique skill phrases semantically using embeddings."""
    phrases = skill_df["term"].tolist()
    embeddings = embed_phrases(phrases)
    distances = cosine_distances(embeddings)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=linkage
    )
    labels = model.fit_predict(distances)
    skill_df = skill_df.copy()
    skill_df["semantic_cluster"] = labels
    return skill_df


def compute_cluster_demand(
    df: pd.DataFrame,
    skill_df: pd.DataFrame,
    text_col: str = "req_text",
    applicants_col: str = "num_applicants",
    use_semantic: bool = True,
) -> pd.DataFrame:
    """Compute demand and supply metrics per semantic cluster."""
    group_col = "semantic_cluster"
    records = []
    for cluster, group in skill_df.groupby(group_col):
        pattern = "|".join(re.escape(t) for t in group["term"])
        mask = df[text_col].str.contains(pattern, regex=True, na=False)
        postings = df[mask]
        if postings.empty:
            continue
        records.append(
            {
                "cluster": int(cluster),
                "demand": int(mask.sum()),
                "avg_applicants": float(postings[applicants_col].mean()),
                "ratio": int(mask.sum()) / postings[applicants_col].mean(),
            }
        )
    return pd.DataFrame(records)
