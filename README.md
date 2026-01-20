# Transcribe Tool

Local audio/video transcription using Whisper. Produces TXT, SRT, VTT, or JSON
from a single command and runs fully offline.

## Requirements

- Python 3.9+
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

- The script prefers `faster-whisper` and falls back to `openai-whisper` if
  needed.
- Output is written next to the input unless `--output` is provided.
