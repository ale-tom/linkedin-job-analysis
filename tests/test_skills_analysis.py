import pandas as pd
import numpy as np

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


from skills_analysis import (  # noqa
    _normalise,  # noqa
    filter_by_ontologies,  # noqa
    replace_keywords_with_language,  # noqa
    load_and_parse_requirements,  # noqa
    extract_noun_phrases,  # noqa
    filter_phrases_by_df,  # noqa
    map_phrases_to_clusters,  # noqa
)  # noqa


def test_normalise():
    assert _normalise("  Hello, World!  ") == "hello world"
    assert _normalise("C++ & C#") == "c c"
    assert _normalise("Multiple   Spaces") == "multiple spaces"


def test_filter_by_ontologies_exact():
    cleaned = [["Python", "Unknown Skill"]]
    ontos = {"prog": ["python"]}
    result = filter_by_ontologies(cleaned, ontos)
    assert result == [["Python"]]


def test_replace_keywords_with_language():
    data = [["r", "I write code in go daily", "scala"]]
    updated = replace_keywords_with_language(data)
    assert "r language" in updated[0]
    assert any("r language" in item.lower() for item in updated[0])
    assert any("go language" in item.lower() for item in updated[0])


def test_load_and_parse_requirements(tmp_path):
    df = pd.DataFrame({"requirements": ['["SkillA", "SkillB"]', np.nan, "Just text"]})
    fp = tmp_path / "req.csv"
    df.to_csv(fp, index=False)
    parsed = load_and_parse_requirements(str(fp))
    # First row combines list items
    assert parsed.loc[0, "req_text"] == "skilla skillb"
    # NaN becomes empty string
    assert parsed.loc[1, "req_text"] == ""
    # Plain text lowercased
    assert parsed.loc[2, "req_text"] == "just text"


def test_extract_noun_phrases():
    texts = ["The quick brown fox jumps over the lazy dog."]
    phrases = extract_noun_phrases(texts)
    # Expect at least two noun phrases
    assert any("quick brown fox" in p for p in phrases[0])
    assert any("lazy dog" in p for p in phrases[0])


def test_filter_phrases_by_df():
    doc_phrases = [["x", "y"], ["y", "z"]]
    # With min_df=0.5, lower_thresh=1, upper_thresh=1.6 -> 'y' count=2 drops
    filtered = filter_phrases_by_df(doc_phrases, min_df=0.5, max_df=0.8)
    assert filtered == [["x"], ["z"]]


def test_map_phrases_to_clusters():
    cleaned = [["regression", "other"]]
    ontos = {"ml": ["regression"]}
    df = map_phrases_to_clusters(cleaned, ontos, fuzzy_threshold=0.8)
    # Should map 'regression' to 'ml', 'other' to 'Other'
    recs = df.set_index("term")["cluster"].to_dict()
    assert recs.get("regression") == "ml"
    assert recs.get("other") == "Other"
