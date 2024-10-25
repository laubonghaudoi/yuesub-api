import os
import torch
from transformers import AutoModelForMaskedLM, BertTokenizerFast
from funasr_onnx import SenseVoiceSmall, Fsmn_vad
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_sensevoice_onnx(model_name):
    if os.path.exists("models/{}".format(model_name)):
        return

    model_dir = "iic/SenseVoiceSmall"
    SenseVoiceSmall(model_dir, batch_size=1, quantize=True, cache_dir="models")


def export_vad_onnx(model_name):
    if os.path.exists("models/{}".format(model_name)):
        return

    Fsmn_vad(
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        batch_size=1,
        quantize=True,
        cache_dir="models",
    )


def get_dummy_input(seq_length=512):
    input_ids = torch.tensor([[i for i in range(seq_length)]], dtype=torch.long)
    attention_mask = torch.tensor([[1 for i in range(seq_length)]], dtype=torch.long)
    token_type_ids = torch.tensor(
        [
            [0 for i in range(int(seq_length / 2))]
            + [1 for i in range(seq_length - int(seq_length / 2))]
        ],
        dtype=torch.long,
    )
    return input_ids, attention_mask, token_type_ids


def export_bert_onnx(model_name):
    model_dir = "models/{}".format(model_name)
    onnx_path = os.path.join(model_dir, "model.onnx")

    if os.path.exists(onnx_path):
        return

    os.makedirs(model_dir, exist_ok=True)

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model.eval()
    dummy_inputs = get_dummy_input(512)  # 16000 * 20

    # export tokenizer
    tokenizer.save_pretrained(model_dir)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            # verbose=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "token_type_ids": {0: "batch", 1: "seq_len"},
                "output": {0: "batch"},
            },
        )
    quantize_dynamic(
        onnx_path,
        onnx_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    export_sensevoice_onnx("iic/SenseVoiceSmall")
    export_vad_onnx("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    export_bert_onnx("hon9kon9ize/bert-large-cantonese")
