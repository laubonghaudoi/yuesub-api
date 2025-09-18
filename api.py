import json
import os
import flask
import utils
from transcriber import AutoTranscriber

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = utils.download_youtube_audio(flask.request.json["video_id"])

    if audio_file is None:
        return flask.Response(response="\n", status=400, mimetype="application/json")

    transcriber = AutoTranscriber(
        corrector="opencc",
        use_denoiser=False,
        with_punct=False,
        offset_in_seconds=0.0,
        max_length_seconds=30.0,
        merge_gap_ms=200,
    )
    transcribe_results = transcriber.transcribe(audio_file)

    os.remove(audio_file)

    if len(transcribe_results) == 0:
        return flask.Response(response="\n", status=400, mimetype="application/json")

    srt_text = utils.to_srt(transcribe_results)

    return flask.Response(response=json.dumps(srt_text), status=200, mimetype="application/json")
