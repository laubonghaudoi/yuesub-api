import os
import argparse
import logging
from pathlib import Path
import utils

logging.basicConfig(level=logging.INFO)


def main():
    # python cli.py your_audio_file.mp3 --output-dir output
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    logging.info("Transcribing %s", args.audio_file)

    transcribe_results = utils.transcribe(args.audio_file)

    if len(transcribe_results) == 0:
        logging.error("No transcriptions found")

    srt_text = utils.to_srt(transcribe_results)
    filename = args.audio_file.split("/")[-1].split(".")[0]
    output_dir = Path(args.output_dir)
    filename = output_dir.joinpath(f"{filename}.srt")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(srt_text)

    logging.info("Transcription saved to %s.srt", filename)


if __name__ == "__main__":
    main()
