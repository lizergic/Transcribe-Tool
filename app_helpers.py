from shutil import which

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
