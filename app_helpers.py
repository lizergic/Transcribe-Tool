import os
import tempfile

from shutil import which

from transcribe import (
    transcribe_with_faster_whisper,
    transcribe_with_openai_whisper,
    write_txt,
    write_srt,
    write_vtt,
    write_json,
)

QUALITY_MAP = {
    "Fast (lower quality)": "tiny",
    "Balanced (recommended)": "small",
    "Best (slower)": "large-v3",
}

QUALITY_CHOICES = list(QUALITY_MAP.keys())
QUALITY_DEFAULT = "Balanced (recommended)"

FORMAT_CHOICES = [
    ("TXT (plain text)", "txt"),
    ("SRT (subtitles)", "srt"),
    ("VTT (web subtitles)", "vtt"),
    ("JSON (structured)", "json"),
]

FORMAT_LABELS = [label for label, _ in FORMAT_CHOICES]
FORMAT_DEFAULT = "TXT (plain text)"
FORMAT_MAP = dict(FORMAT_CHOICES)

LANGUAGE_OPTIONS = [
    ("Auto-detect", None),
    ("en — English", "en"),
    ("es — Spanish", "es"),
    ("fr — French", "fr"),
    ("de — German", "de"),
    ("ja — Japanese", "ja"),
    ("zh — Chinese", "zh"),
    ("ko — Korean", "ko"),
    ("pt — Portuguese", "pt"),
    ("it — Italian", "it"),
    ("ru — Russian", "ru"),
    ("ar — Arabic", "ar"),
    ("hi — Hindi", "hi"),
]

LANGUAGE_CHOICES = [label for label, _ in LANGUAGE_OPTIONS]
LANGUAGE_DEFAULT = "Auto-detect"
LANGUAGE_MAP = dict(LANGUAGE_OPTIONS)


def quality_to_model(label: str) -> str:
    return QUALITY_MAP.get(label, "small")


def format_label_to_ext(label: str) -> str:
    return FORMAT_MAP.get(label, "txt")


def language_choice_to_code(label: str) -> str | None:
    return LANGUAGE_MAP.get(label)


def check_ffmpeg() -> str | None:
    """Return None if ffmpeg is available, or an error message if not."""
    if which("ffmpeg") is None:
        return (
            "ffmpeg not found on PATH. Please install it:\n"
            "  Windows (winget): winget install --id Gyan.FFmpeg -e\n"
            "  macOS (brew): brew install ffmpeg\n"
            "  Linux (apt): sudo apt-get install ffmpeg"
        )
    return None


WRITERS = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def handle_transcribe(
    file_path: str | None,
    quality: str,
    fmt_label: str,
    language_label: str,
) -> tuple[str, str | None]:
    """Run transcription and return (preview_text, output_file_path)."""
    if not file_path:
        return "Please upload an audio or video file first.", None

    ffmpeg_err = check_ffmpeg()
    if ffmpeg_err:
        return ffmpeg_err, None

    model = quality_to_model(quality)
    fmt = format_label_to_ext(fmt_label)
    lang = language_choice_to_code(language_label)

    try:
        segments = transcribe_with_faster_whisper(file_path, model, lang)
    except Exception as err:
        if "No module named" in str(err):
            try:
                segments = transcribe_with_openai_whisper(file_path, model, lang)
            except Exception as err2:
                return (
                    f"Transcription failed. Install dependencies:\n"
                    f"  pip install -r requirements.txt\n"
                    f"Also ensure ffmpeg is installed and on PATH.\n"
                    f"Details: {err2}",
                    None,
                )
        else:
            return f"Transcription failed: {err}", None

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base}.{fmt}")

    writer = WRITERS[fmt]
    writer(segments, out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        preview = f.read()

    return preview, out_path
