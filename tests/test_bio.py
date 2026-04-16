import pytest
from src.data.bio import (
    tokenize_words,
    char_offsets_to_word_bio,
    word_bio_to_char_offsets,
    char_offsets_to_token_bio,
    token_bio_to_char_offsets,
    LABEL2ID,
)
from src.data.corpus import SISpan


# ---------------------------------------------------------------------------
# tokenize_words
# ---------------------------------------------------------------------------

def test_tokenize_words_basic():
    tokens = tokenize_words("hello world")
    assert len(tokens) == 2
    assert tokens[0] == ("hello", 0, 5)
    assert tokens[1] == ("world", 6, 11)


def test_tokenize_words_empty():
    assert tokenize_words("") == []


def test_tokenize_words_preserves_offsets():
    text = "  spaced   out  "
    tokens = tokenize_words(text)
    for word, start, end in tokens:
        assert text[start:end] == word


# ---------------------------------------------------------------------------
# Word-level BIO round-trip
# ---------------------------------------------------------------------------

def test_word_bio_all_outside():
    text = "nothing to see here"
    result = char_offsets_to_word_bio(text, [])
    assert all(tag == 'O' for _, tag, _, _ in result)


def test_word_bio_full_span():
    text = "this is propaganda"
    spans = [SISpan(start=0, end=len(text))]
    result = char_offsets_to_word_bio(text, spans)
    tags = [tag for _, tag, _, _ in result]
    assert tags[0] == 'B'
    assert all(t == 'I' for t in tags[1:])


def test_word_bio_round_trip(simple_article):
    tagged = char_offsets_to_word_bio(simple_article.text, simple_article.si_spans)
    recovered = word_bio_to_char_offsets(tagged)
    assert len(recovered) == len(simple_article.si_spans)


def test_word_bio_no_spans_gives_no_recovered():
    text = "no propaganda here at all"
    tagged = char_offsets_to_word_bio(text, [])
    recovered = word_bio_to_char_offsets(tagged)
    assert recovered == []


# ---------------------------------------------------------------------------
# Token-level BIO (transformer)
# ---------------------------------------------------------------------------

def test_token_bio_special_tokens_ignored():
    spans = []
    # Simulate [CLS] text [SEP] with (0,0) offsets for special tokens
    offset_mapping = [(0, 0), (0, 4), (5, 9), (0, 0)]
    labels = char_offsets_to_token_bio(spans, offset_mapping, text_len=9)
    assert labels[0]  == -100
    assert labels[-1] == -100


def test_token_bio_propaganda_span():
    text = "bad text"
    spans = [SISpan(start=0, end=8)]
    offset_mapping = [(0, 0), (0, 3), (4, 8), (0, 0)]
    labels = char_offsets_to_token_bio(spans, offset_mapping, text_len=len(text))
    assert labels[1] == LABEL2ID['B']
    assert labels[2] == LABEL2ID['I']


def test_token_bio_round_trip():
    text = "this is propaganda text here"
    spans = [SISpan(start=8, end=18)]
    offset_mapping = [(0, 0), (0, 4), (5, 7), (8, 18), (19, 23), (24, 28), (0, 0)]
    labels = char_offsets_to_token_bio(spans, offset_mapping, text_len=len(text))
    recovered = token_bio_to_char_offsets(labels, offset_mapping)
    assert len(recovered) == 1
    assert recovered[0][0] == 8
