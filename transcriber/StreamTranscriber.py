import logging
import os
import re
from typing import List, Literal, Union, Generator, Iterator
import inspect

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
from transcriber.Transcriber import Transcriber

logger = logging.getLogger(__name__)


class StreamTranscriber(Transcriber):
    """
    StreamTranscriber class

    """

    def transcribe(
        self,
        audio_file: str,
    ) -> Generator[TranscribeResult, None, None]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_file (str): Path to audio file

        Returns:
            Generator[TranscribeResult]: Generator of transcription results
        """
        speech, sr = librosa.load(audio_file, sr=self.sr)

        if self.use_denoiser:
            logger.info("Denoising speech...")
            speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000, filter="kaiser_best", parallel=True)

        logger.info("Segmenting speech...")
        vad_segments = self._segment_speech(speech)

        if not vad_segments:
            return []

        pgb_vad_segments = tqdm(
            enumerate(vad_segments), total=len(vad_segments), desc="Transcribing"
        )

        result_generator = self._process_segments(speech, pgb_vad_segments)
        for result in self._convert_to_traditional_chinese(result_generator):
            pgb_vad_segments.set_description(result.text)
            yield result

    def _process_segments(
        self, speech: np.ndarray, pgb_vad_segments: Iterator
    ) -> Generator[TranscribeResult, None, None]:
        """Process each speech segment"""
        speech_lengths = len(speech)

        for _, segment in pgb_vad_segments:
            speech_j, _ = self._slice_padding_audio_samples(
                speech, speech_lengths, [[segment]]
            )

            stt_results = self._asr(speech_j[0])
            timestamp_offset = ((segment[0] * 16) / 16_000) + self.offset_in_seconds

            if not stt_results:
                continue

            for result in stt_results:
                result.start_time += timestamp_offset
                result.end_time += timestamp_offset

                yield result

    def _convert_to_traditional_chinese(
        self, results: Iterator[TranscribeResult]
    ) -> Generator[TranscribeResult, None, None]:
        """Convert simplified Chinese to traditional Chinese"""
        if not results:
            return results

        corrector = Corrector(self.corrector)

        for result in results:
            result.text = corrector.correct(result.text)
            yield result
