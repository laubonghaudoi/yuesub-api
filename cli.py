import argparse
import logging
import os
from pathlib import Path

from utils import to_srt

from transcriber import AutoTranscriber

# Configure logging first, before any imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override any existing logger configurations
)


logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"]


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
        default=True,
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
    # AutoTranscriber is the only supported transcriber; opencc is always used.
    parser.add_argument(
        "--max-length",
        type=float,
        default=30.0,
        help="Maximum length of each segment in seconds",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=int,
        default=200,
        help="Merge adjacent VAD segments if the pause is shorter than this many milliseconds (0 to disable)",
    )
    parser.add_argument(
        "--verbose", help="Increase output verbosity", action="store_true", default=True
    )

    parser.add_argument(
        "--offset_in_seconds",
        help="Offset in seconds to adjust the start time of the transcription",
        type=float,
        default=0,
    )
    args = parser.parse_args()

    try:
        # Initialize transcriber once for all files
        transcriber = AutoTranscriber(
            corrector="opencc",
            use_denoiser=args.denoise,
            with_punct=args.punct,
            offset_in_seconds=args.offset_in_seconds,
            max_length_seconds=args.max_length,
            merge_gap_ms=args.merge_gap_ms,
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
                logger.error("Error transcribing %s: %s", audio_file, e)
                continue

    except Exception as e:
        logger.error("Error during initialization: %s", e)
        raise


if __name__ == "__main__":
    main()
