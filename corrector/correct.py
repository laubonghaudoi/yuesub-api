from itertools import product
from typing import Union

import opencc

from .LanguageModel import LanguageModel
from .BertModel import BertModel

converter = opencc.OpenCC("s2hk.json")


def correct(text: str, t2s_char_dict: dict, corrector: str) -> str:
    """
    Correct the output text using either a language model or OpenCC
    Args:
        text: Input text to correct
        t2s_char_dict: Dictionary mapping traditional to simplified characters
        lm_model: Either 'opencc' or a LanguageModel instance
    Returns:
        Corrected text string
    """
    text = text.strip()
    if not text:  # Early return for empty string
        return text

    if corrector == "opencc":
        return opencc_correct(text)

    elif corrector == "bert":
        bert_model = BertModel("./models/hon9kon9ize/bert-large-cantonese")

        return lm_correct(text, t2s_char_dict, bert_model)
    else:
        raise ValueError("corrector should be either 'opencc' or 'bert'")


def lm_correct(text: str, t2s_char_dict: dict, lm_model: LanguageModel) -> str:
    # Get candidates for each character
    char_candidates = [t2s_char_dict.get(char, [char]) for char in text]

    # If no characters need correction, return original
    if all(len(candidates) == 1 for candidates in char_candidates):
        return text

    # Generate all possible combinations
    text_candidates = ["".join(comb) for comb in product(*char_candidates)]

    if not text_candidates:  # Safeguard against empty candidates
        return text

    # Find the candidate with minimum perplexity score
    return min(text_candidates, key=lambda t: lm_model.perplexity(t))


def opencc_correct(text: str) -> str:
    """
    Convert text using OpenCC
    Args:
        text: Input text to convert
        config: OpenCC configuration
    Returns:
        Converted text string
    """

    return converter.convert(text)
