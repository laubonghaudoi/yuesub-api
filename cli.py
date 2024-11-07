import argparse
import logging
import os
from pathlib import Path

from utils import to_srt

from transcriber.AutoTranscriber import AutoTranscriber
from transcriber.Transcriber import Transcriber

# Configure logging first, before any imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing logger configurations
)


logger = logging.getLogger(__name__)

MODEL_DIRS = [
    "models/iic/SenseVoiceSmall",
    "models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "models/denoiser.onnx",
]


def check_models():
    """Check if all required models are downloaded"""
    for model_dir in MODEL_DIRS:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model not found: {model_dir}\n"
                "Please run `python download_models.py` first"
            )
    if not os.path.exists("models/hon9kon9ize/bert-large-cantonese"):
        logger.info(
            "models/hon9kon9ize/bert-large-cantonese not found, only OpenCC corrector is available",
        )


def save_transcription(srt_text: str, audio_file: str, output_dir: str):
    """Save transcription to SRT file"""
    filename = Path(audio_file).stem
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    srt_path = output_path / f"{filename}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    logger.info("Transcription saved to %s", srt_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT subtitles")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument(
        "--punct", help="Whether to keep punctuation", action="store_true"
    )
    parser.add_argument(
        "--denoise", help="Whether to denoise the audio", action="store_true"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for SRT files",
    )

    args = parser.parse_args()

    try:
        logger.info("Checking models")
        check_models()

        logger.info("Transcribing %s", args.audio_file)

        transcriber = Transcriber(
            corrector="opencc", use_denoiser=args.denoise, with_punct=args.punct
        )
        transcribe_results = transcriber.transcribe(args.audio_file)

        if not transcribe_results:
            logger.error("No transcriptions found")
            return

        srt_text = to_srt(transcribe_results)
        save_transcription(srt_text, args.audio_file, args.output_dir)

    except Exception as e:
        logger.error("Error during transcription: %s", str(e))
        raise


if __name__ == "__main__":
    main()
