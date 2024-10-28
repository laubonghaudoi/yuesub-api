from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import BertTokenizerFast

from .LanguageModel import LanguageModel


@dataclass
class OnnxInferenceResult:
    model_inference_time: List[int]
    optimized_model_path: str


class BertModel(LanguageModel):
    """
    BertModel class
    """

    def __init__(self, model_name="bert-base-chinese"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        self.model = self.create_model_for_provider(
            "{}/model.onnx".format(model_name), provider
        )

    def get_loss(self, text: str) -> float:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        vocab_size = len(self.tokenizer.get_vocab())
        labels = model_inputs.input_ids.to(self.device)
        inputs_onnx = {k: v.to(self.device).detach().numpy() for k, v in model_inputs.items()}
        predictions = self.model.run(None, inputs_onnx)
        lm_logits = torch.from_numpy(predictions[0]).to(self.device)
        loss_fct = torch.nn.CrossEntropyLoss().to(self.device)  # -100 index = padding token
        masked_lm_loss = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
        loss = masked_lm_loss.item()

        return loss

    def perplexity(self, text: str) -> float:
        loss = self.get_loss(text)
        perplexity = np.exp(loss)

        return perplexity.item()
