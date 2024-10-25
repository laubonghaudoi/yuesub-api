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

## Usage

1. export models to ONNX format, it would download the model weights and export to ONNX format in models folder

```bash
$ python export_onnx.py
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
