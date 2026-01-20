import argparse
import json
import os
import sys
from datetime import timedelta
from shutil import which


def format_ts_srt(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ts_vtt(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_txt(segments, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg["text"].strip()
            if text:
                f.write(text + "\n")


def write_json(segments, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def write_srt(segments, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = format_ts_srt(seg["start"])
            end = format_ts_srt(seg["end"])
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")


def write_vtt(segments, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_ts_vtt(seg["start"])
            end = format_ts_vtt(seg["end"])
            text = seg["text"].strip()
            if not text:
                continue
            f.write(f"{start} --> {end}\n{text}\n\n")


def derive_output_path(input_path: str, fmt: str) -> str:
    base, _ = os.path.splitext(input_path)
    return f"{base}.{fmt}"


def transcribe_with_faster_whisper(
    input_path: str, model_name: str, language: str | None
):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        input_path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    out = []
    for seg in segments:
        out.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return out


def transcribe_with_openai_whisper(
    input_path: str, model_name: str, language: str | None
):
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(input_path, language=language)
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in result.get("segments", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text locally."
    )
    parser.add_argument("input", help="Path to audio or video file")
    parser.add_argument(
        "--model",
        default="small",
        help="Model name (e.g. tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en, es). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to input name with chosen extension.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if which("ffmpeg") is None:
        print(
            "ffmpeg not found on PATH. Install it, then re-run.\n"
            "Windows (winget): winget install --id Gyan.FFmpeg -e\n"
            "Verify: ffmpeg -version",
            file=sys.stderr,
        )
        return 2

    out_path = args.output or derive_output_path(args.input, args.format)

    try:
        segments = transcribe_with_faster_whisper(
            args.input, args.model, args.language
        )
    except Exception as err:
        if "No module named" in str(err):
            try:
                segments = transcribe_with_openai_whisper(
                    args.input, args.model, args.language
                )
            except Exception as err2:
                print(
                    "Transcription failed. Install dependencies:\n"
                    "  pip install -r requirements.txt\n"
                    "Also ensure ffmpeg is installed and on PATH.\n"
                    f"Details: {err2}",
                    file=sys.stderr,
                )
                return 1
        else:
            print(f"Transcription failed: {err}", file=sys.stderr)
            return 1

    if args.format == "txt":
        write_txt(segments, out_path)
    elif args.format == "json":
        write_json(segments, out_path)
    elif args.format == "srt":
        write_srt(segments, out_path)
    elif args.format == "vtt":
        write_vtt(segments, out_path)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
