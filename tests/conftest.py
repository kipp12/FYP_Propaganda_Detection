"""
Shared fixtures for all tests.
Uses small synthetic Article objects so tests run without disk access.
"""
import pytest
from src.data.corpus import Article, SISpan, TCSpan


@pytest.fixture
def simple_article():
    """Single article with two non-overlapping propaganda spans."""
    text = "This is normal text. Lies and fear are spread here. More normal text."
    return Article(
        article_id="001",
        text=text,
        si_spans=[SISpan(start=21, end=50), SISpan(start=0, end=4)],
        tc_spans=[
            TCSpan(technique="Loaded_Language", start=21, end=50),
            TCSpan(technique="Doubt",           start=0,  end=4),
        ],
    )


@pytest.fixture
def articles():
    """Small collection of articles for split/model tests."""
    texts = [
        "Propaganda span one is right here in this sentence for testing.",
        "Another article with loaded language right here to test with.",
        "A third article, this one has no propaganda spans at all here.",
        "Fourth article with appeal to fear present in this specific part.",
        "Fifth article with name calling embedded somewhere in this text.",
    ]
    si_spans = [
        [SISpan(start=0, end=20)],
        [SISpan(start=24, end=38)],
        [],
        [SISpan(start=20, end=37)],
        [SISpan(start=22, end=34)],
    ]
    tc_spans = [
        [TCSpan(technique="Loaded_Language",        start=0,  end=20)],
        [TCSpan(technique="Loaded_Language",        start=24, end=38)],
        [],
        [TCSpan(technique="Appeal_to_fear-prejudice", start=20, end=37)],
        [TCSpan(technique="Name_Calling,Labeling",  start=22, end=34)],
    ]
    return [
        Article(article_id=str(i), text=t, si_spans=s, tc_spans=tc)
        for i, (t, s, tc) in enumerate(zip(texts, si_spans, tc_spans))
    ]
