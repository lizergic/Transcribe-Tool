import gradio as gr

from app_helpers import (
    QUALITY_CHOICES,
    QUALITY_DEFAULT,
    FORMAT_LABELS,
    FORMAT_DEFAULT,
    LANGUAGE_CHOICES,
    LANGUAGE_DEFAULT,
    handle_transcribe,
)


def transcribe_click(file, quality, fmt, language, progress=gr.Progress()):
    if not file:
        return "Please upload an audio or video file first.", None
    progress(0.0, desc="Loading model...")
    progress(0.3, desc="Transcribing...")
    preview, out_path = handle_transcribe(file, quality, fmt, language)
    progress(0.9, desc="Writing output...")
    progress(1.0, desc="Done")
    return preview, out_path


with gr.Blocks(title="Transcript Tool") as demo:
    gr.Markdown(
        "# 🎙️ Transcript Tool\n"
        "Local audio/video transcription powered by Whisper"
    )

    file_input = gr.File(
        label="Upload audio or video file",
        file_types=["audio", "video"],
    )

    quality = gr.Dropdown(
        choices=QUALITY_CHOICES,
        value=QUALITY_DEFAULT,
        label="Quality",
    )

    with gr.Accordion("Advanced options", open=False):
        fmt = gr.Dropdown(
            choices=FORMAT_LABELS,
            value=FORMAT_DEFAULT,
            label="Output Format",
        )
        language = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value=LANGUAGE_DEFAULT,
            label="Language",
        )

    btn = gr.Button("Transcribe", variant="primary")

    preview = gr.Textbox(
        label="Transcript Preview",
        interactive=False,
        lines=12,
        buttons=["copy"],
    )
    download = gr.File(label="Download", interactive=False)

    btn.click(
        fn=transcribe_click,
        inputs=[file_input, quality, fmt, language],
        outputs=[preview, download],
    )


if __name__ == "__main__":
    demo.launch()
