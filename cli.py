import argparse
import logging
import os
from pathlib import Path

# Configure logging first, before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing logger configurations
)

from transcribe import to_srt, transcribe

logger = logging.getLogger(__name__)

model_dirs = [
    "models/hon9kon9ize/bert-large-cantonese",
    "models/iic/SenseVoiceSmall",
    "models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "models/denoiser.onnx",
]


def main():
    # python cli.py your_audio_file.mp3 --output-dir output
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str)
    parser.add_argument(
        "--punct", help="Whether to keep punctuation", action="store_true"
    )
    parser.add_argument(
        "--denoise", help="Whether to denoise the audio", action="store_true"
    )
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    # check if all the models are downloaded
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            logger.error(
                "Model not found, please run `python download_models.py` first"
            )
            return

    logger.info("Transcribing %s", args.audio_file)

    transcribe_results = transcribe(
        args.audio_file, 16_000, use_denoiser=args.denoise, with_punct=args.punct
    )

    if len(transcribe_results) == 0:
        logger.error("No transcriptions found")

    srt_text = to_srt(transcribe_results)

    # Save transcription to srt file
    filename = args.audio_file.split("/")[-1].split(".")[0]
    output_dir = Path(args.output_dir)
    filename = output_dir.joinpath(f"{filename}.srt")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(srt_text)

    logger.info("Transcription saved to %s.srt", filename)


if __name__ == "__main__":
    main()
