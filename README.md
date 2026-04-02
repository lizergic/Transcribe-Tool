# Transcribe Tool

Local audio/video transcription using Whisper. Produces TXT, SRT, VTT, or JSON
and runs fully offline. Use the **CLI** for scripting or the **web GUI** for a
point-and-click experience.

## Requirements

- Python 3.10+
- ffmpeg on PATH

## Install

```bash
pip install -r requirements.txt
```

ffmpeg:

- Windows (winget): `winget install --id Gyan.FFmpeg -e`
- macOS (brew): `brew install ffmpeg`
- Linux (apt): `sudo apt-get install ffmpeg`

## Usage

```bash
python transcribe.py INPUT [--model MODEL] [--language LANG]
                           [--format {txt,srt,vtt,json}] [--output PATH]
```

## GUI

Launch the web interface:

```bash
python app.py
```

Opens a browser tab at `http://localhost:7860`.

1. Upload an audio or video file
2. Pick a quality level — Fast (`tiny`), Balanced (`small`), or Best (`large-v3`)
3. Click **Transcribe**

Advanced options (output format, language) are available under the expandable
accordion. The transcript appears in a preview pane with a copy button, and a
download link is provided for the output file.

## Flags

- `--model`: Whisper model name. Examples: `tiny`, `base`, `small`, `medium`,
  `large-v3`. Default: `small`.
- `--language`: Language code like `en`, `es`. Omit to auto-detect.
- `--format`: Output format: `txt`, `srt`, `vtt`, `json`. Default: `txt`.
- `--output`: Output path. Defaults to input name with chosen extension.

## Examples

```bash
# Basic transcription to TXT
python transcribe.py sample.mp3

# Force language and output SRT
python transcribe.py sample.mp4 --language en --format srt

# Use a larger model and custom output file
python transcribe.py sample.wav --model medium --output out/transcript.txt
```

## Notes

- Both the CLI and GUI prefer `faster-whisper` and fall back to `openai-whisper`
  if needed.
- CLI output is written next to the input unless `--output` is provided.
  GUI output is available via the download link.
- Recommend using the tiny/Fast model for larger audio files or low CPU
  availability.
