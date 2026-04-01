# Gradio GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio web GUI (`app.py`) that wraps the existing CLI transcription tool, targeting non-technical users.

**Architecture:** Single new file `app.py` imports from the unchanged `transcribe.py`. A helper module `app_helpers.py` contains the mapping logic and transcription handler, keeping the Gradio UI definition thin. Tests cover the helper logic.

**Tech Stack:** Python, Gradio, faster-whisper/openai-whisper (existing)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `app_helpers.py` | Create | Quality/format/language mappings, transcription handler function |
| `app.py` | Create | Gradio UI layout and wiring (imports handler from `app_helpers.py`) |
| `tests/test_app_helpers.py` | Create | Unit tests for mapping logic and handler |
| `requirements.txt` | Modify | Add `gradio` |
| `README.md` | Modify | Add GUI usage section |

---

### Task 1: Add Gradio Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

```
faster-whisper
openai-whisper
gradio
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "deps: add gradio to requirements"
```

---

### Task 2: Quality/Format/Language Mapping Helpers (TDD)

**Files:**
- Create: `app_helpers.py`
- Create: `tests/test_app_helpers.py`

- [ ] **Step 1: Write failing tests for mapping functions**

Create `tests/test_app_helpers.py`:

```python
from app_helpers import (
    QUALITY_CHOICES,
    FORMAT_CHOICES,
    LANGUAGE_CHOICES,
    quality_to_model,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_app_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app_helpers'`

- [ ] **Step 3: Implement mapping helpers**

Create `app_helpers.py`:

```python
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


def quality_to_model(label: str) -> str:
    return QUALITY_MAP.get(label, "small")


def format_label_to_ext(label: str) -> str:
    mapping = {l: ext for l, ext in FORMAT_CHOICES}
    return mapping.get(label, "txt")


def language_choice_to_code(label: str) -> str | None:
    mapping = {l: code for l, code in LANGUAGE_OPTIONS}
    return mapping.get(label)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_app_helpers.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app_helpers.py tests/test_app_helpers.py
git commit -m "feat: add quality/format/language mapping helpers with tests"
```

---

### Task 3: Transcription Handler (TDD)

**Files:**
- Modify: `app_helpers.py`
- Modify: `tests/test_app_helpers.py`

- [ ] **Step 1: Write failing tests for the handler**

Append to `tests/test_app_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/test_app_helpers.py -v -k "handle_transcribe"`
Expected: FAIL with `ImportError: cannot import name 'handle_transcribe'`

- [ ] **Step 3: Implement the handler**

Append to `app_helpers.py`:

```python
import os
import tempfile

from transcribe import (
    transcribe_with_faster_whisper,
    transcribe_with_openai_whisper,
    write_txt,
    write_srt,
    write_vtt,
    write_json,
)

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
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `python -m pytest tests/test_app_helpers.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app_helpers.py tests/test_app_helpers.py
git commit -m "feat: add transcription handler with whisper fallback and tests"
```

---

### Task 4: Gradio UI Layout

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py with the Gradio interface**

Create `app.py`:

```python
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
        show_copy_button=True,
    )
    download = gr.File(label="Download", interactive=False)

    btn.click(
        fn=transcribe_click,
        inputs=[file_input, quality, fmt, language],
        outputs=[preview, download],
    )


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Smoke test — verify it launches**

Run: `python app.py`
Expected: Gradio prints `Running on local URL: http://127.0.0.1:7860` and opens a browser tab. Verify the layout matches the spec: file upload, quality dropdown, collapsed Advanced accordion, Transcribe button, empty preview and download areas. Close with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio web GUI for transcription"
```

---

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add GUI section to README**

After the existing `## Usage` section's CLI content, add a new section. Insert before `## Flags`:

```markdown
## GUI

Launch the web interface:

```bash
python app.py
```

Opens a browser tab at `http://localhost:7860`. Upload a file, pick a quality level,
and click Transcribe. Advanced options (output format, language) are available under
the expandable section.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add GUI usage instructions to README"
```

---

### Task 6: Manual End-to-End Verification

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Launch the GUI and test with a real file**

Run: `python app.py`

Verify:
1. Page loads with title, file upload, quality dropdown, collapsed Advanced
2. Expanding Advanced shows format (default TXT) and language (default Auto-detect)
3. Upload a short audio file, select "Fast (lower quality)", click Transcribe
4. Progress phases appear: "Loading model..." → "Transcribing..." → "Writing output..." → "Done"
5. Transcript text appears in preview area
6. Download link appears and downloads a `.txt` file

- [ ] **Step 3: Verify CLI still works**

Run: `python transcribe.py --help`
Expected: Normal CLI help output, unchanged.

- [ ] **Step 4: Commit any fixes if needed, then tag completion**

```bash
git log --oneline -5
```

Verify all commits are present. Done.
