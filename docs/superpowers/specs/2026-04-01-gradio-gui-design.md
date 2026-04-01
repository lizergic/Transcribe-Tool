# Transcript Tool — Gradio GUI Design Spec

## Goal

Add a polished, non-technical-friendly web GUI to the existing CLI transcription tool using Gradio. The GUI wraps the existing `transcribe.py` functionality without modifying it.

## Target User

Non-technical users who want to transcribe audio/video files locally. The interface should be self-explanatory with no CLI knowledge required.

## Architecture

```
app.py (new)  ──imports──▶  transcribe.py (unchanged)
```

- **`app.py`**: New file. Defines the Gradio interface and a transcription handler that bridges Gradio inputs to existing transcription functions.
- **`transcribe.py`**: No changes. `app.py` imports and calls its functions directly (`transcribe_with_faster_whisper`, `transcribe_with_openai_whisper`, `write_txt`, `write_srt`, `write_vtt`, `write_json`).
- **`requirements.txt`**: Add `gradio` as a dependency.

## UI Layout

### Default View

1. **Header**: App title ("Transcript Tool") and subtitle ("Local audio/video transcription powered by Whisper")
2. **File upload**: `gr.File` — drag-and-drop or click to browse. Accepts audio/video formats.
3. **Quality dropdown**: `gr.Dropdown` with three options:
   - "Fast (lower quality)" → maps to `tiny`
   - "Balanced (recommended)" → maps to `small` (default)
   - "Best (slower)" → maps to `large-v3`
4. **Advanced options**: `gr.Accordion(open=False)` containing:
   - **Output format**: `gr.Dropdown` — TXT (default), SRT, VTT, JSON
   - **Language**: `gr.Dropdown` — "Auto-detect" (default) + common language codes (en, es, fr, de, ja, zh, ko, pt, it, ru, ar, hi)
5. **Transcribe button**: `gr.Button("Transcribe", variant="primary")`

### Results View (after transcription)

6. **Progress**: `gr.Progress` with phase-level status text:
   - Phase 1: "Loading model..."
   - Phase 2: "Transcribing..."
   - Phase 3: "Writing output..."
7. **Transcript preview**: `gr.Textbox(interactive=False)` — read-only text area showing the transcription result. For JSON/SRT/VTT, shows the raw formatted output.
8. **Download**: `gr.File` — download link for the output file.

## Data Flow

1. User uploads file → Gradio stores it in a temp path
2. User clicks Transcribe →
   a. Validate ffmpeg is on PATH (reuse `which("ffmpeg")` check from `transcribe.py`)
   b. Map quality label to model name
   c. Progress: "Loading model..."
   d. Call `transcribe_with_faster_whisper(temp_path, model, language)` with fallback to `transcribe_with_openai_whisper()`
   e. Progress: "Transcribing..."  (set before the call — the transcription itself is the long step)
   f. Progress: "Writing output..."
   g. Write output using the appropriate `write_*()` function to a temp file
   h. Read the output file content for the preview textbox
3. Return preview text + output file path to Gradio components

## Error Handling

- **Missing ffmpeg**: Show a friendly `gr.Warning` with install instructions (Windows/macOS/Linux), matching the existing CLI message.
- **Missing whisper dependencies**: Show `gr.Warning` with `pip install -r requirements.txt` instructions.
- **No file uploaded**: Disable or grey out the Transcribe button until a file is present.
- **Transcription failure**: Catch exceptions, display error in the preview textbox with a clear message.

## Quality-to-Model Mapping

| Label | Model | Use Case |
|-------|-------|----------|
| Fast (lower quality) | `tiny` | Quick drafts, large files |
| Balanced (recommended) | `small` | Default, good accuracy |
| Best (slower) | `large-v3` | Maximum accuracy |

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `app.py` | Create | Gradio GUI application |
| `requirements.txt` | Edit | Add `gradio` |

## Launch

```bash
python app.py
```

Opens a local web browser tab. Gradio serves on `http://localhost:7860` by default.

## Out of Scope

- Batch file processing
- Inline transcript editing
- Real-time/streaming transcription preview
- User accounts or persistent history
- Custom Gradio theming (use default theme)
