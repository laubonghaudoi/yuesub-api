import logging
import time
from typing import List, Literal

import librosa
import numpy as np
from funasr import AutoModel
from resampy.core import resample
from tqdm.auto import tqdm

from denoiser import denoiser
from silero_vad import load_silero_vad, get_speech_timestamps
from transcriber.Transcriber import Transcriber, TranscribeResult

logger = logging.getLogger(__name__)


class AutoTranscriber(Transcriber):
    """
    Transcriber class that uses FunASR's AutoModel for VAD and ASR
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize models
        self.vad_model = load_silero_vad()
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=None,  # We'll handle VAD separately
            punc_model=None,
            ban_emo_unks=True,
        )

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
        # Load and preprocess audio
        speech, sr = librosa.load(audio_file, sr=None)

        if self.use_denoiser:
            logger.info("Denoising speech...")
            speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000,
                              filter="kaiser_best", parallel=True)

        # Get VAD segments
        logger.info("Segmenting speech...")

        start_time = time.time()
        vad_results = get_speech_timestamps(
            speech,
            self.vad_model,
            sampling_rate=16_000,
            # max_speech_duration_s=self.max_length_seconds,
        )
        logger.info("VAD took %.2f seconds", time.time() - start_time)

        if not vad_results:
            return []

        # Process each segment
        results = []

        start_time = time.time()
        for segment in tqdm(vad_results, desc="Transcribing"):
            segment_audio = speech[segment["start"] : segment["end"]]

            # Get ASR results for segment
            asr_result = self.asr_model.generate(
                input=segment_audio, language="yue", use_itn=self.with_punct, disable_pbar=True
            )

            if not asr_result:
                continue
            start_time = max(0, segment['start'] / 16_000.0 + self.offset_in_seconds)
            end_time = segment['end'] / 16_000.0  + self.offset_in_seconds

            # Convert ASR result to TranscribeResult format
            segment_result = TranscribeResult(
                text=asr_result[0]["text"],
                start_time=start_time,  # Convert ms to seconds
                end_time=end_time,
            )
            results.append(segment_result)

        logger.info("ASR took %.2f seconds", time.time() - start_time)
        # Apply Chinese conversion if needed
        start_time = time.time()
        results = self._postprocessing(results)
        logger.info("Conversion took %.2f seconds", time.time() - start_time)

        return results
