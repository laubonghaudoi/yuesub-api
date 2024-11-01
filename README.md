# Cantonese Subtitle Transcript Service

## Introduction

This service API used SenseVoice, VAD and Bert model to generate Cantonese subtitle transcript for audio file.

This is version only support Youtube video URL.

1. Download audio file from Youtube video URL
2. Use VAD model to split audio file into small audio clips
3. Use SenseVoice model to generate Cantonese subtitle transcript and timestamp for each audio clip
4. Since the output of SenseVoice model is Simplified Chinese, we use OpenCC to convert it to Traditional Chinese and then use Bert to correct the translation
5. Generate SRT file for the Cantonese subtitle transcript

## Models

All model are exporting as ONNX format.

1. SenseVoice: iic/SenseVoiceSmall(on ModelScope)
2. VAD: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch(on ModelScope)
3. Bert: hon9kon9ize/bert-large-cantonese

## Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

1. export models to ONNX format, it would download the model weights and export to ONNX format in models folder

```bash
$ python download_models.py
```

2. run the cli

```bash
$ python cli.py your_audio.mp3 --output-dir output
```

or in batch

```bash
for file in $(ls *.mp3); do python cli.py $file --output-dir output; done
```

3. or run the web API service

```bash
$ python app.py
```

## 使用教程

### 準備工作

將本 repo clone 落本地後，跑下面嘅命令嚟安裝依賴，然後下載必需嘅模型：

```bash
pip install -r requirements.txt
python download_models.py
```

跟住準備好你需要轉寫嘅音頻文件，如果你想下載 YouTube 片音頻，可以裝 `pip install yt-dlp` 然後跑下面嘅命令嚟下載

```bash
# 呢條命令係單純下載音頻，冇視頻嘅，如果想要下載埋視頻就刪咗個 -f ba 佢
yt-dlp -f ba https://youtu.be/rIBD6A4lnLQ
```

### 轉寫

單獨轉寫一個文件可以直接跑

```bash
python cli.py audio.mp3 --output_dir output
```

如果要轉寫晒路經下所有 mp3 文件，可以跑

```bash
for file in $(ls *.webm); do python cli.py $file --output-dir output --punct; done
```
