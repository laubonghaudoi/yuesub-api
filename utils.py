import os
import re
import psutil
from pytubefix import YouTube
from funasr_onnx import Fsmn_vad_online, SenseVoiceSmall
from torchaudio.pipelines import MMS_FA as bundle
import librosa
import numpy as np
import tempfile
from resampy.core import resample
import torch
from typing import List, Union, Literal
from transformers import BertTokenizerFast
from dataclasses import dataclass
from denoiser import denoiser
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)
from pysrt import SubRipFile
from pysrt import SubRipItem
from pysrt import SubRipTime
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

asr_model = SenseVoiceSmall("./models/iic/SenseVoiceSmall", batch_size=1, quantize=True)
vad_model = Fsmn_vad_online(
    "./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", batch_size=1, quantize=True
)

LABELS = [
    asr_model.tokenizer.sp.IdToPiece(i)
    for i in range(asr_model.tokenizer.sp.piece_size())
]
special_labels = [
    label for label in LABELS if label.startswith("<|") and label.endswith("|>")
]

special_token_ids = [
    asr_model.tokenizer.sp.PieceToId(i)
    for i in ["<s>", "</s>", "<unk>", "<pad>"] + special_labels
]
aligner = bundle.get_aligner()
SAMPLE_RATE = 16000


def load_dict(
    t2s_dict_file="./data/STCharacters.txt",
    jyutping_dict_file="./data/jyut6ping3.chars.dict.tsv",
    chars_freq_dict_file="./data/chars_freq.tsv",
):
    char_jyutping_dict = {}
    jyutping_char_dict = {}
    chars_freq = {}
    t2s_char_dict = {}

    with open(chars_freq_dict_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                char, freq = line.split("\t")
                chars_freq[char] = float(freq)

    with open(jyutping_dict_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    char, jyutping = line.split("\t")

                    if char in char_jyutping_dict:
                        char_jyutping_dict[char].append(jyutping)
                    else:
                        char_jyutping_dict[char] = [jyutping]

                    if jyutping in jyutping_char_dict:
                        jyutping_char_dict[jyutping].append(char)
                    else:
                        jyutping_char_dict[jyutping] = [char]
                except:
                    print(line)

    # sort jyutping_char_dict by char freq
    jyutping_char_dict = {
        k: sorted(
            v, key=lambda x: chars_freq[x] if x in chars_freq else 0, reverse=True
        )
        for k, v in jyutping_char_dict.items()
    }

    with open(t2s_dict_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sc, tc = line.split("\t")
                tc = tc.split(" ")
                t2s_char_dict[sc] = tc
    # patch for 晒, 咁
    t2s_char_dict["晒"] = ["晒", "曬"]
    t2s_char_dict["咁"] = ["咁", "噉"]
    t2s_char_dict["旧"] = t2s_char_dict["旧"] + ["嚿"]

    return t2s_char_dict, char_jyutping_dict, jyutping_char_dict, chars_freq


def download_youtube_audio(video_id: str) -> str:
    urls = "https://www.youtube.com/watch?v={}".format(video_id)

    try:
        # https://github.com/JuanBindez/pytubefix/issues/242#issuecomment-2369067929
        vid = YouTube(urls, "MWEB")

        if vid.title is None:
            return None

        audio_download = vid.streams.get_audio_only()
        audio_download.download(
            mp3=True,
            filename=video_id,
            output_path=tempfile.gettempdir(),
            skip_existing=True,
        )
        audio_file = tempfile.gettempdir() + "/" + video_id + ".mp3"

        return audio_file

    except Exception as e:
        print(e)
        return None


def slice_padding_audio_samples(speech, speech_lengths, vad_segments):
    speech_list = []
    speech_lengths_list = []
    for i, segment in enumerate(vad_segments):
        bed_idx = int(segment[0][0] * 16)
        end_idx = min(int(segment[0][1] * 16), speech_lengths)
        speech_i = speech[bed_idx:end_idx]
        speech_lengths_i = end_idx - bed_idx
        speech_list.append(speech_i)
        speech_lengths_list.append(speech_lengths_i)

    return speech_list, speech_lengths_list


class TranscribeResult:
    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"TranscribeResult(text={self.text}, start_time={self.start_time}, end_time={self.end_time})"

    def __repr__(self):
        return str(self)


def asr(
    wav_content: Union[str, np.ndarray, List[str]],
    language="yue",
    textnorm: Union[Literal["withitn", "woitn"]] = "withitn",
    with_punct=False,
) -> List["TranscribeResult"]:
    language_input = language
    textnorm_input = textnorm
    language_list, textnorm_list = asr_model.read_tags(language_input, textnorm_input)
    waveform = asr_model.load_data(
        wav_content, asr_model.frontend.opts.frame_opts.samp_freq
    )
    feats, feats_len = asr_model.extract_feat(waveform)
    _language_list = language_list[0:1]
    _textnorm_list = textnorm_list[0:1]
    segments = []
    results = []

    ctc_logits, _ = asr_model.infer(
        feats,
        feats_len,
        np.array(_language_list, dtype=np.int32),
        np.array(_textnorm_list, dtype=np.int32),
    )
    ctc_logits = torch.from_numpy(ctc_logits).float()
    ratio = waveform[0].shape[0] / ctc_logits.size(1) / SAMPLE_RATE
    # support batch_size=1 only currently
    x = ctc_logits[0]
    # log_softmax
    x = torch.nn.functional.log_softmax(x, dim=-1)
    yseq = x.argmax(dim=-1)
    yseq = torch.unique_consecutive(yseq, dim=-1)

    mask = yseq != asr_model.blank_id
    preds = yseq[mask]
    token_spans = aligner(ctc_logits[0], preds.unsqueeze(0))[0]
    results = _process_segments(token_spans, ratio, with_punct)

    return results


def _process_segments(
    token_spans,
    ratio,
    with_punct,
    punct_labels=",?!。，；？！",
    sep_labels="?!。；？！",
):
    """
    Processes token spans into segments with optional punctuation handling.

    Args:
        token_spans (list): A list of token spans, where each span contains a token, start time, and end time.
        ratio (float): A ratio to adjust the start and end times of the segments.
        with_punct (bool): A flag indicating whether to include punctuation in the segments.
        punct_labels (str, optional): A string of punctuation characters to consider. Defaults to "?!。，；？！".
        sep_labels (str, optional): A string of separator characters to consider. Defaults to "?!。；？

    Returns:
        list: A list of TranscribeResult objects, each containing text, start time, and end time.
    """
    segments = []

    for token_span in token_spans:
        label = asr_model.tokenizer.sp.IdToPiece(token_span.token)

        if token_span.token in special_token_ids:
            continue

        segments.append(
            TranscribeResult(
                text=label,
                start_time=token_span.start * ratio,
                end_time=token_span.end * ratio,
            )
        )

    results = []
    start_idx = 0
    end_idx = 1

    while end_idx < len(segments):
        current_segment = segments[end_idx]

        if current_segment.text in sep_labels:
            curr_end_idx = end_idx

            if with_punct:
                curr_end_idx += 1

            text = "".join(
                [segment.text for segment in segments[start_idx:curr_end_idx]]
            )

            if not with_punct:
                text = re.sub(f"[{punct_labels}]", " ", text)

            results.append(
                TranscribeResult(
                    text=text,
                    start_time=segments[start_idx].start_time,
                    end_time=segments[end_idx].end_time,
                )
            )
            start_idx = end_idx + 1

        end_idx += 1

    return results


# abstract class for language model
class LanguageModel:
    def perplexity(self, text: str) -> float:
        raise NotImplementedError()

    def get_loss(self, text: str) -> torch.Tensor:
        raise NotImplementedError()


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

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


@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str


class BertModel(LanguageModel):
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = create_model_for_provider(
            "{}/model.onnx".format(model_name), "CPUExecutionProvider"
        )

    def get_loss(self, text: str) -> float:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        vocab_size = len(self.tokenizer.get_vocab())
        labels = model_inputs.input_ids
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        predictions = self.model.run(None, inputs_onnx)
        lm_logits = predictions[0]
        lm_logits = torch.from_numpy(lm_logits)
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1))
        loss = masked_lm_loss.item()

        return loss

    def perplexity(self, text: str) -> float:
        loss = self.get_loss(text)
        perplexity = np.exp(loss)

        return perplexity.item()


t2s_char_dict, char_jyutping_dict, jyutping_char_dict, chars_freq = load_dict()

bert_model = BertModel("./models/hon9kon9ize/bert-large-cantonese")


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


def transcribe(audio_file: str) -> List["TranscribeResult"]:
    speech, sr = librosa.load(audio_file)
    speech, new_sr = denoiser(speech, sr)

    if new_sr != 16_000:
        speech = resample(speech, new_sr, 16_000, filter="kaiser_best", parallel=True)

    logger.info("Segmenting speech")

    res = vad_model(speech)
    vadsegments = res[0]
    n = len(vadsegments)
    speech_lengths = len(speech)
    results = []

    logger.info("Number of segments: %d", n)

    if not n:
        return []

    for j, _ in tqdm(enumerate(range(n)), total=n, desc="Transcribing"):
        speech_j, speech_lengths_j = slice_padding_audio_samples(
            speech, speech_lengths, [[vadsegments[j]]]
        )
        stt_results = asr(speech_j[0])
        timestamp_offset = ((vadsegments[j][0] * 16) / 16_000) - 0.1

        if len(stt_results) < 1:
            continue

        # add timestamp offset
        for result in stt_results:
            result.start_time += timestamp_offset
            result.end_time += timestamp_offset

        results.extend(stt_results)

    # convert to Traditional Chinese
    for result in tqdm(
        results, total=len(results), desc="Converting to Traditional Chinese"
    ):
        result.text = corrector(result.text, t2s_char_dict, bert_model)

    return results


def to_srt(results: List["TranscribeResult"]) -> str:
    srt = SubRipFile()

    for i, t in enumerate(results):
        start = SubRipTime(seconds=t.start_time)
        end = SubRipTime(seconds=t.end_time)
        item = SubRipItem(index=i, start=start, end=end, text=t.text)
        srt.append(item)

    temp_file = tempfile.gettempdir() + "/output.srt"
    srt.save(temp_file)

    with open(temp_file, "r", encoding="utf-8") as f:
        srt_text = f.read()

    os.remove(temp_file)

    return srt_text
