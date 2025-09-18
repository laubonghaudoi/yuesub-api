# 粵文字幕生成器 Cantonese Subtitle Transcript Service

[English](#introduction)

呢個係粵文字幕生成器，輸入音頻文件（.mp3 .wav .webm .flac 等等）輸出.srt 字幕文件。

粵語轉寫用 [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) 配合 Silero VAD 做切分。字幕文字以 OpenCC 進行繁簡轉換及規則修正。

## 使用教程

### 準備工作

將本 repo clone 落本地後，跑下面嘅命令嚟安裝依賴，然後下載必需嘅模型：

```bash
apt install ffmpeg
pip install -r requirements.txt
```

跟住準備好你需要轉寫嘅音頻文件，如果你想下載 YouTube 片音頻，可以裝 `pip install yt-dlp` 然後跑下面嘅命令嚟下載

```bash
# 呢條命令係單純下載音頻，冇視頻嘅，如果想要下載埋視頻就刪咗個 -f ba 佢
yt-dlp -f ba https://youtu.be/rIBD6A4lnLQ
```

### 轉寫

跑下面嘅命令，將你嘅音頻文件轉寫成字幕；默認使用 OpenCC 規則式修正。

單獨轉寫一個文件可以直接跑

```bash
python cli.py audio.mp3 --output_dir output
```

如果唔特指某個文件而係成個路經，就會自動轉寫晒路經下所有嘅音頻：

```bash
# 自動轉寫晒所有 audio/ 入面嘅音頻
python cli.py ./audio/ --output_dir output
```

## Introduction

This service uses SenseVoice and VAD for transcription and OpenCC for traditional Chinese conversion and rule-based fixes to generate Cantonese subtitles.

This version supports local files via CLI and a simple web UI; the API includes a YouTube helper to download audio if needed.

1. Download audio file from Youtube video URL
2. Use VAD model to split audio file into small audio clips
3. Use SenseVoice model to generate Cantonese subtitle transcript and timestamp for each audio clip
4. Since the output of SenseVoice model is Simplified Chinese, we use OpenCC to convert it to Traditional Chinese and apply rule-based Cantonese fixes
5. Generate SRT file for the Cantonese subtitle transcript

## Models

Models are loaded dynamically via FunASR and Silero VAD at runtime; no ONNX export is required.

## Prerequisites

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

## Usage

### Prerequisites

Ensure ffmpeg and Python dependencies are installed. Models are downloaded automatically by FunASR/Silero on first run.

You can run the following command to download a YouTube audio. Make sure you have yt-dlp installed by `pip install yt-dlp`.

```bash
# download audio file from youtube video url, if you want to download video as well, remove -f ba
yt-dlp -f ba https://youtu.be/rIBD6
```

### Transcribe

Run the CLI (OpenCC corrector is enabled by default). You can increase context and merge small pauses for better quality.

single file transcription can be run directly

```bash
$ python cli.py your_audio.mp3 --output-dir output --max-length 30 --merge-gap-ms 200
```

or in batch

```bash
# Auto transcribe all audio files under the audio/ directory
python cli.py ./audio/ --output_dir output
```

or run the web API service

```bash
$ python app.py
```
