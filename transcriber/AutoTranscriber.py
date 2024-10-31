import logging
import re
from typing import List, Literal

import librosa
import numpy as np
from funasr import AutoModel
from resampy.core import resample
from tqdm.auto import tqdm

from corrector.Corrector import Corrector
from denoiser import denoiser
from transcriber.TranscribeResult import TranscribeResult

logger = logging.getLogger(__name__)


class AutoTranscriber:
    """
    Transcriber class that uses FunASR's AutoModel for VAD and ASR
    """

    def __init__(self, corrector: Literal["opencc", "bert", None] = None, use_denoiser=False, with_punct=True, sr=16000):
        self.corrector = corrector
        self.use_denoiser = use_denoiser
        self.with_punct = with_punct
        self.sr = sr
        
        # Initialize models
        self.vad_model = AutoModel(model="fsmn-vad")
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=None,  # We'll handle VAD separately
            punc_model="ct-punc" if with_punct else None,
            ban_emo_unks=True
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
        speech, sr = librosa.load(audio_file, sr=self.sr)

        if self.use_denoiser:
            logger.info("Denoising speech...")
            speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000,
                            filter="kaiser_best", parallel=True)

        # Get VAD segments
        logger.info("Segmenting speech...")
        vad_results = self.vad_model.generate(input=speech)
        if not vad_results or not vad_results[0]["value"]:
            return []
        
        vad_segments = vad_results[0]["value"]

        # Process each segment
        results = []
        for segment in tqdm(vad_segments, desc="Transcribing"):
            start_sample = int(segment[0] * 16)  # Convert ms to samples
            end_sample = int(segment[1] * 16)
            segment_audio = speech[start_sample:end_sample]
            
            # Get ASR results for segment
            asr_result = self.asr_model.generate(input=segment_audio, language="yue", use_itn=True)
            if not asr_result:
                continue
                
            # Convert ASR result to TranscribeResult format
            segment_result = TranscribeResult(
                text=asr_result[0]["text"],
                start_time=segment[0] / 1000.0,  # Convert ms to seconds
                end_time=segment[1] / 1000.0
            )
            results.append(segment_result)

        # Apply Chinese conversion if needed
        results = self._convert_to_traditional_chinese(results)

        return results

    def _convert_to_traditional_chinese(
        self,
        results: List[TranscribeResult]
    ) -> List[TranscribeResult]:
        """Convert simplified Chinese to traditional Chinese"""
        if not results or not self.corrector:
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
