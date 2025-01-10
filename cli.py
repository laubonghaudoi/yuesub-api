import argparse
import logging
import os
from pathlib import Path

from utils import to_srt

from transcriber import StreamTranscriber, OnnxTranscriber, AutoTranscriber

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

SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"]


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
    parser = argparse.ArgumentParser(
        description="Transcribe Cantonese audio to SRT subtitles"
    )
    parser.add_argument("input_path", type=str, help="Path to audio file or directory")
    parser.add_argument(
        "--punct",
        help="Whether to keep punctuation",
        action="store_true",
        default=False,
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
    parser.add_argument(
        "--onnx",
        help="Use ONNX runtime for transcription",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stream",
        help="Streaming the transcription preview while processing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--corrector",
        type=str,
        default="opencc",
        help="Corrector type: opencc or bert",
        choices=["opencc", "bert"],
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=3.0,
        help="Maximum length of each segment in seconds",
    )
    parser.add_argument(
        "--verbose", help="Increase output verbosity", action="store_true", default=True
    )

    parser.add_argument(
        "--offset_in_seconds",
        help="Offset in seconds to adjust the start time of the transcription",
        type=float,
        default=-0.25,
    )
    args = parser.parse_args()

    if args.stream == True and args.funasr == True:
        raise ValueError("Cannot use both stream and funasr")

    transcriber_class = [StreamTranscriber, AutoTranscriber, OnnxTranscriber][
        0 if args.stream == True else 2 if args.onnx == True else 1
    ]

    try:
        if transcriber_class == OnnxTranscriber: 
            logger.info("Checking models")
            check_models()

        # Initialize transcriber once for all files
        transcriber = transcriber_class(
            corrector="opencc",
            use_denoiser=args.denoise,
            with_punct=args.punct,
            offset_in_seconds=args.offset_in_seconds,
            max_length_seconds=args.max_length,
        )

        input_path = Path(args.input_path)

        if input_path.is_file():
            # Single file mode
            audio_files = [str(input_path)]
        else:
            # Directory mode
            audio_files = [
                str(f)
                for f in input_path.glob("*")
                if f.suffix.lower() in SUPPORTED_FORMATS
            ]
            if not audio_files:
                logger.error("No audio files found in directory: %s", input_path)
                return
            logger.info("Found %d audio files to process", len(audio_files))

        # Process each audio file
        for audio_file in audio_files:
            try:
                logger.info("Transcribing %s", audio_file)
                transcribe_results = transcriber.transcribe(audio_file)

                if not transcribe_results:
                    logger.warning("No transcriptions found for %s", audio_file)
                    continue

                srt_text = to_srt(transcribe_results)
                save_transcription(srt_text, audio_file, args.output_dir)

            except Exception as e:
                logger.error("Error transcribing %s: %s", audio_file, str(e))
                continue

    except Exception as e:
        logger.error("Error during initialization: %s", str(e))
        raise


if __name__ == "__main__":
    main()
