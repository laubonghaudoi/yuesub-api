import logging
from typing import Generator, Iterator

import librosa
import numpy as np
from resampy.core import resample
from tqdm.auto import tqdm

from denoiser import denoiser
from transcriber.Transcriber import TranscribeResult
from transcriber.OnnxTranscriber import OnnxTranscriber

logger = logging.getLogger(__name__)


class StreamTranscriber(OnnxTranscriber):
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

        for result_generator in self._process_segments(speech, pgb_vad_segments):
            for result in self._postprocessing([result_generator]):
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
