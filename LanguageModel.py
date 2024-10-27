import psutil
import torch
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)


# abstract class for language model
class LanguageModel:
    def perplexity(self, text: str) -> float:
        raise NotImplementedError()

    def get_loss(self, text: str) -> torch.Tensor:
        raise NotImplementedError()

    def create_model_for_provider(
        self, model_path: str, provider: str
    ) -> InferenceSession:
        """
        Create an ONNX model for a specific provider
        :param model_path: path to the model
        :param provider: provider name
        :return: InferenceSession
        """
        assert (
            provider in get_all_providers()
        ), f"provider {provider} not found, {get_all_providers()}"

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()

        return session
