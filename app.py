import logging
import tempfile

import gradio as gr

from transcriber import AutoTranscriber
from utils import to_srt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def transcribe_audio(audio_path):
    """Process audio file and return SRT content and preview text"""
    try:
        transcriber = AutoTranscriber(
            corrector="opencc",
            use_denoiser=False,
            with_punct=False
        )

        transcribe_results = transcriber.transcribe(audio_path)

        if not transcribe_results:
            return None, "無字幕生成， 可能係檢測唔到語音。"

        # Generate SRT text for both preview and download
        srt_text = to_srt(transcribe_results)

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.srt', encoding='utf-8') as tmp:
            tmp.write(srt_text)
            return tmp.name, srt_text

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return None, f"Error: {str(e)}"


def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 粵文字幕生成器")
        gr.Markdown(
            "上傳一個音頻文件，撳「生成字幕」，過一陣就會得到 SRT 文件。目前支援格式：.mp3、.wav、.flac、.m4a、.ogg、opus、.webm")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="上傳音頻文件或者錄音")

        with gr.Row():
            generate_btn = gr.Button("生成字幕 SRT 文件", variant="primary", scale=2)

        with gr.Row():
            with gr.Column():
                preview = gr.Textbox(label="預覽生成字幕", lines=10)

            with gr.Column():
                output = gr.File(label="下載 SRT")

        generate_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output, preview]
        )

    return demo


def main():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=8081)


if __name__ == "__main__":
    main()
