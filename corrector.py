from LanguageModel import LanguageModel


def corrector(text: str, t2s_char_dict: dict, lm_model: LanguageModel) -> str:
    text = text.strip()
    char_candidates = []

    if text == "":
        return text

    for char in text:
        if char in t2s_char_dict:
            char_candidates.append(t2s_char_dict[char])
        else:
            char_candidates.append([char])

    # make all possible candidates
    text_candidates = []

    for i, candidates in enumerate(char_candidates):
        if i == 0:
            text_candidates = candidates
        else:
            new_candidates = []
            for c in candidates:
                for t in text_candidates:
                    new_candidates.append(t + c)
            text_candidates = new_candidates

    if len(text_candidates) == 0:
        return text

    # get score of each char with kenlm
    scores = []

    for t in text_candidates:
        scores.append(lm_model.perplexity(t))

    # sort by score
    text_candidates = [
        x for _, x in sorted(zip(scores, text_candidates), key=lambda pair: pair[0])
    ]

    return text_candidates[0]
