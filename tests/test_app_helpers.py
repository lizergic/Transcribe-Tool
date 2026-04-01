from app_helpers import (
    QUALITY_CHOICES,
    FORMAT_CHOICES,
    LANGUAGE_CHOICES,
    quality_to_model,
    format_label_to_ext,
    language_choice_to_code,
)


def test_quality_choices_has_three_options():
    assert len(QUALITY_CHOICES) == 3


def test_quality_to_model_fast():
    assert quality_to_model("Fast (lower quality)") == "tiny"


def test_quality_to_model_balanced():
    assert quality_to_model("Balanced (recommended)") == "small"


def test_quality_to_model_best():
    assert quality_to_model("Best (slower)") == "large-v3"


def test_quality_to_model_default_on_unknown():
    assert quality_to_model("garbage") == "small"


def test_format_choices_contains_all_formats():
    labels = [label for label, _ in FORMAT_CHOICES]
    assert "TXT (plain text)" in labels
    assert "SRT (subtitles)" in labels
    assert "VTT (web subtitles)" in labels
    assert "JSON (structured)" in labels


def test_language_auto_detect_maps_to_none():
    assert language_choice_to_code("Auto-detect") is None


def test_language_english_maps_to_code():
    assert language_choice_to_code("en — English") == "en"


def test_language_spanish_maps_to_code():
    assert language_choice_to_code("es — Spanish") == "es"


def test_format_label_to_ext_txt():
    assert format_label_to_ext("TXT (plain text)") == "txt"


def test_format_label_to_ext_srt():
    assert format_label_to_ext("SRT (subtitles)") == "srt"


def test_format_label_to_ext_default_on_unknown():
    assert format_label_to_ext("garbage") == "txt"


def test_language_choice_to_code_unknown_returns_none():
    assert language_choice_to_code("garbage") is None


import tempfile
import os
from unittest.mock import patch, MagicMock

from app_helpers import handle_transcribe


def test_handle_transcribe_no_file_returns_error():
    preview, dl = handle_transcribe(
        None, "Balanced (recommended)", "TXT (plain text)", "Auto-detect"
    )
    assert "upload" in preview.lower()
    assert dl is None


@patch("app_helpers.which", return_value=None)
def test_handle_transcribe_no_ffmpeg_returns_error(mock_which):
    preview, dl = handle_transcribe(
        "fake.mp3", "Balanced (recommended)", "TXT (plain text)", "Auto-detect"
    )
    assert "ffmpeg" in preview.lower()
    assert dl is None


@patch("app_helpers.which", return_value="/usr/bin/ffmpeg")
@patch("app_helpers.transcribe_with_faster_whisper")
def test_handle_transcribe_success_txt(mock_transcribe, mock_which):
    mock_transcribe.return_value = [
        {"start": 0.0, "end": 1.0, "text": " Hello world"},
        {"start": 1.0, "end": 2.0, "text": " Goodbye"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        tmp_path = f.name
    try:
        preview, dl = handle_transcribe(
            tmp_path, "Fast (lower quality)", "TXT (plain text)", "Auto-detect"
        )
        assert "Hello world" in preview
        assert "Goodbye" in preview
        assert dl is not None
        assert dl.endswith(".txt")
        mock_transcribe.assert_called_once_with(tmp_path, "tiny", None)
    finally:
        os.unlink(tmp_path)
        if dl and os.path.exists(dl):
            os.unlink(dl)


@patch("app_helpers.which", return_value="/usr/bin/ffmpeg")
@patch("app_helpers.transcribe_with_faster_whisper")
def test_handle_transcribe_success_srt(mock_transcribe, mock_which):
    mock_transcribe.return_value = [
        {"start": 0.0, "end": 1.5, "text": " Hello world"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        tmp_path = f.name
    try:
        preview, dl = handle_transcribe(
            tmp_path, "Balanced (recommended)", "SRT (subtitles)", "en — English"
        )
        assert "-->" in preview  # SRT timestamp format
        assert "Hello world" in preview
        assert dl.endswith(".srt")
        mock_transcribe.assert_called_once_with(tmp_path, "small", "en")
    finally:
        os.unlink(tmp_path)
        if dl and os.path.exists(dl):
            os.unlink(dl)


@patch("app_helpers.which", return_value="/usr/bin/ffmpeg")
@patch(
    "app_helpers.transcribe_with_faster_whisper",
    side_effect=ImportError("No module named 'faster_whisper'"),
)
@patch("app_helpers.transcribe_with_openai_whisper")
def test_handle_transcribe_falls_back_to_openai(
    mock_openai, mock_faster, mock_which
):
    mock_openai.return_value = [
        {"start": 0.0, "end": 1.0, "text": " Fallback text"},
    ]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        tmp_path = f.name
    try:
        preview, dl = handle_transcribe(
            tmp_path, "Balanced (recommended)", "TXT (plain text)", "Auto-detect"
        )
        assert "Fallback text" in preview
        mock_openai.assert_called_once()
    finally:
        os.unlink(tmp_path)
        if dl and os.path.exists(dl):
            os.unlink(dl)


@patch("app_helpers.which", return_value="/usr/bin/ffmpeg")
@patch(
    "app_helpers.transcribe_with_faster_whisper",
    side_effect=ImportError("No module named 'faster_whisper'"),
)
@patch(
    "app_helpers.transcribe_with_openai_whisper",
    side_effect=RuntimeError("Everything is broken"),
)
def test_handle_transcribe_both_backends_fail(
    mock_openai, mock_faster, mock_which
):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        tmp_path = f.name
    try:
        preview, dl = handle_transcribe(
            tmp_path, "Balanced (recommended)", "TXT (plain text)", "Auto-detect"
        )
        assert "failed" in preview.lower() or "error" in preview.lower()
        assert dl is None
    finally:
        os.unlink(tmp_path)
