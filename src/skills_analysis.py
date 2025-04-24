import ast
import re
from collections import Counter
from typing import List

import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# Load spaCy model for noun chunk extraction and embeddings
nlp = spacy.load("en_core_web_md")


def embed_phrases(phrases: List[str]) -> np.ndarray:
    return np.vstack([nlp(p).vector for p in phrases])


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


def map_phrases_to_clusters(cleaned_phrases: List[List[str]]) -> pd.DataFrame:
    """Aggregate cleaned phrases into a DataFrame of unique terms and frequencies."""
    phrase_counts = Counter(p for phrases in cleaned_phrases for p in phrases)
    skill_df = pd.DataFrame(
        [(term, freq) for term, freq in phrase_counts.items()], columns=["term", "freq"]
    )
    return skill_df


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


def plot_skill_frequency(skill_df: pd.DataFrame, top_n: int = 10) -> None:
    """Plot bar chart of top N skills by number of postings."""
    top = skill_df.sort_values("freq", ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    plt.barh(top["term"][::-1], top["freq"][::-1])
    plt.xlabel("Number of Postings")
    plt.title(f"Top {top_n} Skills by Postings")
    plt.tight_layout()
    plt.show()


def plot_cooccurrence_heatmap(
    df: pd.DataFrame, key_terms: List[str], text_col: str = "req_text"
) -> None:
    """Display a heatmap of term co-occurrence for selected key_terms."""
    cooc = pd.DataFrame(0, index=key_terms, columns=key_terms)
    for text in df[text_col].dropna().str.lower():
        present = [t for t in key_terms if t in text]
        for i in present:
            for j in present:
                cooc.loc[i, j] += 1
    plt.figure(figsize=(7, 6))
    plt.imshow(cooc, aspect="equal")
    plt.xticks(range(len(key_terms)), key_terms, rotation=90)
    plt.yticks(range(len(key_terms)), key_terms)
    plt.colorbar(label="Co-occurrence")
    plt.title("Skill Co-occurrence Heatmap")
    plt.tight_layout()
    plt.show()


def plot_demand_vs_supply(cluster_df: pd.DataFrame) -> None:
    """Scatter plot of cluster demand vs. supply (avg. applicants)."""
    plt.figure(figsize=(6, 6))
    plt.scatter(cluster_df["demand"], cluster_df["avg_applicants"])
    for _, row in cluster_df.iterrows():
        plt.text(row["demand"], row["avg_applicants"], row["cluster"], fontsize=9)
    plt.axhline(cluster_df["avg_applicants"].mean(), linestyle="--")
    plt.axvline(cluster_df["demand"].mean(), linestyle="--")
    plt.xlabel("Demand (Postings)")
    plt.ylabel("Supply (Avg Applicants)")
    plt.title("Cluster Demand vs. Supply")
    plt.tight_layout()
    plt.show()
