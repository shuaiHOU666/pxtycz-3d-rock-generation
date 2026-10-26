"""Frozen SciBERT text encoder used for text conditioning.

The weights are downloaded from the Hugging Face Hub on first use
(``allenai/scibert_scivocab_uncased``); they are *not* shipped with this repository.
"""
from __future__ import annotations

from transformers import AutoModel, AutoTokenizer

DEFAULT_TEXT_ENCODER = "allenai/scibert_scivocab_uncased"


def create_text_encoder(name_or_path: str = DEFAULT_TEXT_ENCODER):
    """Return ``(model, tokenizer)`` for a BERT-style encoder.

    ``name_or_path`` may be a Hub identifier or a local directory.
    """
    model = AutoModel.from_pretrained(name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return model, tokenizer
