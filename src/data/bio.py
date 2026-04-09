import re


# ---------------------------------------------------------------------------
# Word-level BIO (used by LR model)
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> list:
    """
    Split text into non-whitespace tokens, returning (word, start, end) tuples
    where start/end are character offsets into the original text.
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]


def char_offsets_to_word_bio(text: str, si_spans: list) -> list:
    """
    Convert character-level SI span annotations to word-level BIO tags.

    Args:
        text:      raw article text
        si_spans:  list of SISpan objects with .start and .end

    Returns:
        List of (word, tag, char_start, char_end) tuples.
        tag is one of 'B', 'I', 'O'.
    """
    # Build a character-level propaganda mask to handle overlapping spans cleanly
    mask = bytearray(len(text))
    for span in si_spans:
        for i in range(span.start, min(span.end, len(text))):
            mask[i] = 1

    tokens = tokenize_words(text)
    result = []
    prev_tag = 'O'

    for word, start, end in tokens:
        is_prop = any(mask[i] for i in range(start, end))
        if is_prop:
            tag = 'I' if prev_tag in ('B', 'I') else 'B'
        else:
            tag = 'O'
        result.append((word, tag, start, end))
        prev_tag = tag

    return result


def word_bio_to_char_offsets(tagged_tokens: list) -> list:
    """
    Reconstruct character-level span offsets from word-level BIO predictions.

    Args:
        tagged_tokens: list of (word, tag, char_start, char_end)

    Returns:
        List of (start, end) character offset tuples.
    """
    spans = []
    span_start = None
    span_end = None

    for _, tag, start, end in tagged_tokens:
        if tag == 'B':
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start = start
            span_end = end
        elif tag == 'I' and span_start is not None:
            span_end = end
        else:
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start = None
            span_end = None

    if span_start is not None:
        spans.append((span_start, span_end))

    return spans


# ---------------------------------------------------------------------------
# Token-level BIO for transformer models (BERT / RoBERTa)
# ---------------------------------------------------------------------------

def char_offsets_to_token_bio(si_spans: list, offset_mapping: list, text_len: int) -> list:
    """
    Assign BIO tags to transformer subword tokens using the tokenizer's
    offset_mapping. Handles special tokens ([CLS], [SEP], padding) which
    have offset (0, 0) — these are assigned a special label of -100 so
    they are ignored by the loss function.

    Args:
        si_spans:       list of SISpan objects
        offset_mapping: list of (start, end) char positions per token,
                        as returned by a HuggingFace tokenizer with
                        return_offsets_mapping=True
        text_len:       length of the original text (to bounds-check)

    Returns:
        List of integer labels, one per token:
            -100 = special token (ignore in loss)
               0 = O
               1 = B
               2 = I
    """
    # Build character-level mask
    mask = bytearray(text_len)
    for span in si_spans:
        for i in range(span.start, min(span.end, text_len)):
            mask[i] = 1

    labels = []
    prev_label = 0  # O

    for token_start, token_end in offset_mapping:
        # Special tokens have (0, 0) offset — mark as ignore
        if token_start == 0 and token_end == 0:
            labels.append(-100)
            prev_label = 0  # reset BIO continuity at special tokens
            continue

        is_prop = any(mask[i] for i in range(token_start, token_end))
        if is_prop:
            label = 2 if prev_label in (1, 2) else 1  # I=2 if continuing, else B=1
        else:
            label = 0  # O

        labels.append(label)
        prev_label = label

    return labels


def token_bio_to_char_offsets(labels: list, offset_mapping: list) -> list:
    """
    Reconstruct character-level span offsets from transformer token-level
    BIO predictions.

    Args:
        labels:         list of integer labels (0=O, 1=B, 2=I, -100=ignore)
        offset_mapping: list of (start, end) per token

    Returns:
        List of (start, end) character offset tuples.
    """
    spans = []
    span_start = None
    span_end = None

    for label, (tok_start, tok_end) in zip(labels, offset_mapping):
        if label == -100:
            continue
        if label == 1:  # B
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start = tok_start
            span_end = tok_end
        elif label == 2 and span_start is not None:  # I
            span_end = tok_end
        else:  # O
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start = None
            span_end = None

    if span_start is not None:
        spans.append((span_start, span_end))

    return spans


# Numeric label mappings for reference
LABEL2ID = {'O': 0, 'B': 1, 'I': 2}
ID2LABEL = {0: 'O', 1: 'B', 2: 'I'}
