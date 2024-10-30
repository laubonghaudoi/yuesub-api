import logging
import os
import tempfile
from typing import List

from pysrt import SubRipFile, SubRipItem, SubRipTime
from pytubefix import YouTube

from transcriber import TranscribeResult

logger = logging.getLogger(__name__)


def download_youtube_audio(video_id: str) -> str:
    """
    Download audio from YouTube video.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        str: Path to the downloaded audio file.
    """
    urls = "https://www.youtube.com/watch?v={}".format(video_id)

    try:
        # https://github.com/JuanBindez/pytubefix/issues/242#issuecomment-2369067929
        vid = YouTube(urls, "MWEB")

        if vid.title is None:
            return None

        audio_download = vid.streams.get_audio_only()
        audio_download.download(
            mp3=True,
            filename=video_id,
            output_path=tempfile.gettempdir(),
            skip_existing=True,
        )
        audio_file = tempfile.gettempdir() + "/" + video_id + ".mp3"

        return audio_file

    except Exception as e:
        print(e)
        return None


def to_srt(results: List["TranscribeResult"]) -> str:
    """
    Convert the list of TranscribeResult objects into a SRT file
    """
    srt = SubRipFile()

    for i, t in enumerate(results):
        start = SubRipTime(seconds=t.start_time)
        end = SubRipTime(seconds=t.end_time)
        item = SubRipItem(index=i, start=start, end=end, text=t.text)
        srt.append(item)

    temp_file = tempfile.gettempdir() + "/output.srt"
    srt.save(temp_file)

    with open(temp_file, "r", encoding="utf-8") as f:
        srt_text = f.read()

    os.remove(temp_file)

    return srt_text
