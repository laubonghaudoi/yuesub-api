from itertools import product
from typing import Union

from .LanguageModel import LanguageModel


def correct(text: str, t2s_char_dict: dict, lm_model: Union[str, LanguageModel]) -> str:
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

    if isinstance(lm_model, str) and lm_model == "opencc":
        return NotImplementedError("OpenCC is not implemented yet")
    
    if not isinstance(lm_model, LanguageModel):
        raise ValueError("lm_model should be either 'opencc' or a LanguageModel object")

    # Get candidates for each character
    char_candidates = [
        t2s_char_dict.get(char, [char]) for char in text
    ]

    # If no characters need correction, return original
    if all(len(candidates) == 1 for candidates in char_candidates):
        return text

    # Generate all possible combinations
    text_candidates = [''.join(comb) for comb in product(*char_candidates)]
    
    if not text_candidates:  # Safeguard against empty candidates
        return text

    # Find the candidate with minimum perplexity score
    return min(text_candidates, key=lambda t: lm_model.perplexity(t))
