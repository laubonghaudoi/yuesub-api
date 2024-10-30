import logging
import os
import re
from typing import List, Literal, Union

import librosa
import numpy as np
import onnxruntime
import torch
from funasr_onnx import Fsmn_vad_online, SenseVoiceSmall
from funasr_onnx.utils.sentencepiece_tokenizer import SentencepiecesTokenizer
from resampy.core import resample
from torchaudio.pipelines import MMS_FA as bundle
from tqdm.auto import tqdm

from corrector.Corrector import Corrector
from denoiser import denoiser
from transcriber.TranscribeResult import TranscribeResult

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Transcriber class

    """

    def __init__(self, corrector: Literal["opencc", "bert", None] = None, use_denoiser=False, with_punct=True, sr=16000):
        self.corrector = corrector
        self.use_denoiser = use_denoiser
        self.with_punct = with_punct
        self.sr = sr

        self._setup_device()
        self._load_models()
        self._load_tokenizer()
        self._setup_aligner()

    def _setup_device(self):
        """Set up device and providers"""
        self.available_providers = onnxruntime.get_available_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in self.available_providers else "cpu"
        self.providers = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in self.available_providers
            else "CPUExecutionProvider"
        )

        logger.info(f"Using device: {self.device}")
        logger.info(f"Available ONNX providers: {self.available_providers}")
        logger.info(f"Selected providers: {self.providers}")

    def _load_models(self):
        """Load ASR and VAD models"""
        self.asr_model = SenseVoiceSmall(
            "./models/iic/SenseVoiceSmall",
            batch_size=1,
            quantize=True,
            providers=self.providers
        )
        self.vad_model = Fsmn_vad_online(
            "./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            batch_size=1,
            quantize=True,
            providers=self.providers,
        )

    def _load_tokenizer(self):
        """Load and setup tokenizer"""
        self.tokenizer = SentencepiecesTokenizer(
            bpemodel=os.path.join(
                "./models/iic/SenseVoiceSmall",
                "chn_jpn_yue_eng_ko_spectok.bpe.model"
            )
        )
        self.labels = [self.tokenizer.sp.IdToPiece(
            i) for i in range(self.tokenizer.sp.piece_size())]
        self.special_labels = [
            label for label in self.labels
            if label.startswith("<|") and label.endswith("|>")
        ]
        self.special_token_ids = [
            self.tokenizer.sp.PieceToId(i)
            for i in ["<s>", "</s>", "<unk>", "<pad>"] + self.special_labels
        ]

    def _setup_aligner(self):
        """Setup aligner"""
        self.aligner = bundle.get_aligner()
        self.sample_rate = 16000

    def transcribe(
        self,
        audio_file: str,
    ) -> List[TranscribeResult]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_file (str): Path to audio file

        Returns:
            List[TranscribeResult]: List of transcription results
        """
        speech, sr = librosa.load(audio_file, sr=self.sr)

        if self.use_denoiser:
            logger.info("Denoising speech...")
            speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000,
                              filter="kaiser_best", parallel=True)

        logger.info("Segmenting speech...")
        vad_segments = self._segment_speech(speech)

        if not vad_segments:
            return []

        results = self._process_segments(speech, vad_segments)
        results = self._convert_to_traditional_chinese(results)

        return results

    def _segment_speech(self, speech: np.ndarray) -> List:
        """Segment speech using VAD model"""
        res = self.vad_model(speech)
        return res[0]

    def _process_segments(
        self,
        speech: np.ndarray,
        vad_segments: List
    ) -> List[TranscribeResult]:
        """Process each speech segment"""
        speech_lengths = len(speech)
        results = []

        for j, _ in tqdm(
            enumerate(vad_segments),
            total=len(vad_segments),
            desc="Transcribing"
        ):
            speech_j, _ = self._slice_padding_audio_samples(
                speech,
                speech_lengths,
                [[vad_segments[j]]]
            )

            stt_results = self._asr(speech_j[0])
            timestamp_offset = ((vad_segments[j][0] * 16) / 16_000) - 0.1

            if not stt_results:
                continue

            for result in stt_results:
                result.start_time += timestamp_offset
                result.end_time += timestamp_offset

            results.extend(stt_results)

        return results

    def _convert_to_traditional_chinese(
        self,
        results: List[TranscribeResult]
    ) -> List[TranscribeResult]:
        """Convert simplified Chinese to traditional Chinese"""
        if not results:
            return results

        corrector = Corrector(self.corrector)
        if self.corrector == "bert":
            for result in tqdm(
                results,
                total=len(results),
                desc="Converting to Traditional Chinese"
            ):
                result.text = corrector.correct(result.text)
        elif self.corrector == "opencc":
            # Use a special delimiter that won't appear in Chinese text
            delimiter = "|||"
            # Concatenate all texts with delimiter
            combined_text = delimiter.join(result.text for result in results)
            # Convert all text at once
            converted_text = corrector.correct(combined_text)
            # Split back into individual results
            converted_parts = converted_text.split(delimiter)

            # Update results with converted text
            for result, converted in zip(results, converted_parts):
                result.text = converted

        return results

    def _slice_padding_audio_samples(
        self,
        speech: np.ndarray,
        speech_lengths: int,
        vad_segments: List
    ) -> tuple:
        """Slice and pad audio samples based on VAD segments"""
        speech_list = []
        speech_lengths_list = []

        for segment in vad_segments:
            bed_idx = int(segment[0][0] * 16)
            end_idx = min(int(segment[0][1] * 16), speech_lengths)
            speech_i = speech[bed_idx:end_idx]
            speech_lengths_i = end_idx - bed_idx
            speech_list.append(speech_i)
            speech_lengths_list.append(speech_lengths_i)

        return speech_list, speech_lengths_list

    def _asr(
        self,
        wav_content: Union[str, np.ndarray, List[str]],
        language="yue",
        textnorm: Union[Literal["withitn", "woitn"]] = "withitn",
    ) -> List[TranscribeResult]:
        """Perform ASR on audio content

        Args:
            wav_content (Union[str, np.ndarray, List[str]]): Audio content
            language (str): Language code
            textnorm (Union[Literal["withitn", "woitn"]]): Text normalization

        Returns:
            List[TranscribeResult]: List of transcription results
        """
        language_input = language
        textnorm_input = textnorm
        language_list, textnorm_list = self.asr_model.read_tags(
            language_input,
            textnorm_input
        )

        waveform = self.asr_model.load_data(
            wav_content,
            self.asr_model.frontend.opts.frame_opts.samp_freq
        )
        feats, feats_len = self.asr_model.extract_feat(waveform)

        ctc_logits, _ = self.asr_model.infer(
            feats,
            feats_len,
            np.array(language_list[0:1], dtype=np.int32),
            np.array(textnorm_list[0:1], dtype=np.int32),
        )

        ctc_logits = torch.from_numpy(ctc_logits).float().to(self.device)
        ratio = waveform[0].shape[0] / ctc_logits.size(1) / self.sample_rate

        x = ctc_logits[0]
        x = torch.nn.functional.log_softmax(x, dim=-1)
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)

        mask = yseq != self.asr_model.blank_id
        preds = yseq[mask]
        token_spans = self.aligner(ctc_logits[0], preds.unsqueeze(0))[0]

        return self._process_token_spans(token_spans, ratio)

    def _process_token_spans(
        self,
        token_spans,
        ratio: float,
        punct_labels: str = ",?!。，；？！",
        sep_labels: str = "?!。；？！",
    ) -> List[TranscribeResult]:
        """Process token spans into segments"""
        segments = []
        results = []

        for token_span in token_spans:
            label = self.tokenizer.sp.IdToPiece(token_span.token)

            if token_span.token in self.special_token_ids:
                continue

            segments.append(
                TranscribeResult(
                    text=label,
                    start_time=token_span.start * ratio,
                    end_time=token_span.end * ratio,
                )
            )

        start_idx = 0
        end_idx = 1

        while end_idx < len(segments):
            current_segment = segments[end_idx]

            if current_segment.text in sep_labels:
                curr_end_idx = end_idx + 1 if self.with_punct else end_idx

                text = "".join(
                    [segment.text for segment in segments[start_idx:curr_end_idx]]
                )

                if not self.with_punct:
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
