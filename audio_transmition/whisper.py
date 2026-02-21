"""Whisper + pyannote diarization (single file or folder).

Examples:
  set HF_TOKEN=***
  python transcribe_whisper.py --audio "audio transcript/athul1.ogg" --model base --out-dir "audio transcript"
  python transcribe_whisper.py --folder "audio transcript" --model base --out-dir "audio transcript"
"""
import argparse
import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pyannote.audio import Pipeline
import google.generativeai as genai

# Avoid importing this file as the whisper module
import sys
from importlib import import_module
from contextlib import contextmanager


@contextmanager
def import_real_whisper():
    here = Path(__file__).resolve().parent
    original_path = sys.path.copy()
    # Drop current directory so we load the installed whisper package
    sys.path = [p for p in sys.path if Path(p).resolve() != here]
    try:
        yield import_module("whisper")
    finally:
        sys.path = original_path


with import_real_whisper() as whisper:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper + diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="Path to one audio file")
    group.add_argument("--folder", help="Transcribe all audio files in a folder")
    parser.add_argument("--backend", choices=["whisper", "gemini"], default="gemini", help="Transcription backend")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--gemini-model", default="models/gemini-2.5-flash", help="Gemini model name (e.g., models/gemini-2.5-flash)")
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY"), help="API key for Gemini backend")
    parser.add_argument("--language", default=None, help="Language hint (e.g., en, hi, ml); leave empty to auto-detect")
    parser.add_argument("--stereo-split", action="store_true", help="If stereo, left=User1 right=User2 (skip diarization)")
    parser.add_argument("--min-seg-dur", type=float, default=0.8, help="Min diarized segment length (s)")
    parser.add_argument("--merge-gap", type=float, default=0.3, help="Merge same-speaker if gap <= this (s)")
    parser.add_argument("--out-dir", default="audio transcript", help="Output directory for .txt files")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), help="HF token for pyannote (optional, if omitted diarization is skipped)")
    return parser.parse_args()


def load_audio(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32")
    wav = torch.tensor(data)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def gemini_transcribe(audio_np: np.ndarray, sr: int, model, args: argparse.Namespace) -> str:
    """Send a mono float32 segment to Gemini via File Upload API and return transcript."""
    # Write segment to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        sf.write(tmp_path, audio_np, sr, format="WAV")

        # Upload the file
        print(f"Uploading segment to Gemini...")
        audio_file = genai.upload_file(path=tmp_path)
        
        # Wait briefly for processing if necessary
        time.sleep(1)

        prompt = (
            "Please provide a word-for-word transcript of this audio. "
            "There are multiple speakers in this audio. You must diarize the conversation "
            "by labeling each spoken segment with 'User 1:', 'User 2:', etc. "
            "Do not summarize, add any conversational filler, or describe the audio. "
            "Just write the speaker label followed by their spoken words."
        )
        if args.language == "english":
             prompt = (
                "Please provide a word-for-word transcript of this audio, translated into English. "
                "There are multiple speakers in this audio. You must diarize the conversation "
                "by labeling each spoken segment with 'User 1:', 'User 2:', etc. "
                "Do not summarize, add any conversational filler, or describe the audio. "
                "Just write the speaker label followed by their spoken words in English."
            )
        elif args.language:
            prompt = (
                f"Please provide a word-for-word transcript of this audio in {args.language}. "
                "There are multiple speakers in this audio. You must diarize the conversation "
                "by labeling each spoken segment with 'User 1:', 'User 2:', etc. "
                "Do not summarize, add any conversational filler, or describe the audio. "
                "Just write the speaker label followed by their spoken words."
            )

        resp = model.generate_content([
            audio_file,
            prompt,
        ])
        if not resp.text:
            print(f"Gemini returned empty text. Response: {resp}")
        
        # Clean up Gemini file
        genai.delete_file(audio_file.name)
        
        return (resp.text or "").strip()
    except Exception as exc:  # Best-effort logging; keep run going
        print(f"Gemini transcription error: {exc}")
        return ""
    finally:
        # Clean up local file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def diarize_segments(wav: torch.Tensor, sr: int, pipeline: Pipeline, args: argparse.Namespace):
    mono = wav.mean(dim=0)
    diar = pipeline({"waveform": mono.unsqueeze(0), "sample_rate": sr})

    annotation = None
    if hasattr(diar, "itertracks"):
        annotation = diar
    elif isinstance(diar, dict):
        annotation = diar.get("diarization") or diar.get("annotation")
    elif hasattr(diar, "diarization"):
        annotation = diar.diarization
    elif hasattr(diar, "speaker_diarization"):
        annotation = diar.speaker_diarization
    if annotation is None:
        raise RuntimeError("Could not extract diarization annotation")

    raw = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if (turn.end - turn.start) < args.min_seg_dur:
            continue
        raw.append((turn.start, turn.end, speaker))

    raw.sort(key=lambda x: x[0])
    merged = []
    for seg in raw:
        if not merged:
            merged.append(seg)
            continue
        last = merged[-1]
        gap = seg[0] - last[1]
        if seg[2] == last[2] and gap <= args.merge_gap:
            merged[-1] = (last[0], seg[1], last[2])
        else:
            merged.append(seg)
    return [(s, e, spk, None) for s, e, spk in merged]


def transcribe_one(audio_path: Path, model, pipeline: Pipeline, device: torch.device, args: argparse.Namespace):
    wav, sr = load_audio(audio_path)
    segments = []
    mode = "diar"

    if args.stereo_split and wav.shape[0] == 2:
        duration = wav.shape[1] / sr
        segments.append((0.0, duration, "User1", wav[0]))
        segments.append((0.0, duration, "User2", wav[1]))
        mode = "stereo"
    elif pipeline is not None:
        segments = diarize_segments(wav, sr, pipeline, args)
    else:
        duration = wav.shape[1] / sr
        segments.append((0.0, duration, "Speaker", wav.mean(dim=0)))
        mode = "mono_no_diarization"

    mono = wav.mean(dim=0)
    results = []
    for start, end, spk, pre in segments:
        if mode in ("stereo", "mono_no_diarization") and pre is not None:
            audio_t = pre
        else:
            s_i = int(start * sr)
            e_i = int(end * sr)
            audio_t = mono[s_i:e_i]
        if audio_t.numel() == 0:
            continue
        audio_np = audio_t.numpy().astype(np.float32)
        if args.backend == "whisper":
            text = model.transcribe(audio_np, language=args.language, fp16=(device.type == "cuda")).get("text", "").strip()
        else:
            text = gemini_transcribe(audio_np, sr, model, args)
        results.append((start, end, spk, text))

    speaker_map = {}
    user_counter = 1
    if mode == "diar":
        for _, _, spk, _ in results:
            if spk not in speaker_map:
                speaker_map[spk] = f"User{user_counter}"
                user_counter += 1

    lines = []
    for start, end, spk, text in results:
        label = spk if mode in ("stereo", "mono_no_diarization") else speaker_map.get(spk, spk)
        ts = f"[{start:6.2f}-{end:6.2f}]"
        lines.append(f"{ts} {label}: {text}")
    return lines


def main() -> None:
    args = parse_args()
    if not args.hf_token and not args.stereo_split:
        print("Warning: HF_TOKEN not provided, Pyannote speaker diarization will be skipped.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wmodel = None
    gmodel = None
    if args.backend == "whisper":
        wmodel = whisper.load_model(args.model, device=device.type)
    else:
        if not args.google_api_key:
            raise SystemExit("GOOGLE_API_KEY required for Gemini backend (set env or --google-api-key)")
        genai.configure(api_key=args.google_api_key)
        gmodel = genai.GenerativeModel(args.gemini_model)

    pipeline = None
    if not args.stereo_split and args.hf_token:
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token)
            if pipeline:
                pipeline.to(device)
        except Exception as e:
            print(f"Failed to load Pyannote pipeline: {e}\nDiarization will be skipped.")
            pipeline = None

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            raise SystemExit(f"Folder not found: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".ogg", ".opus", ".mp3", ".flac", ".m4a"})
        if not files:
            raise SystemExit(f"No audio files in {folder}")
    else:
        ap = Path(args.audio)
        if not ap.exists():
            raise SystemExit(f"Audio not found: {ap}")
        files = [ap]

    # Resolve output dir without nesting when already inside the target folder.
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        cwd = Path.cwd()
        if out_dir.name == cwd.name:
            out_dir = cwd
        else:
            out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, ap in enumerate(files, 1):
        print(f"\n=== File {idx}/{len(files)}: {ap.name} ===")
        try:
            lines = transcribe_one(ap, wmodel or gmodel, pipeline, device, args)
        except Exception as exc:
            print(f"Error processing {ap.name}: {exc}")
            continue

        for line in lines:
            print(line)
        if not lines:
            lines = ["(no transcript)"]

        out_path = out_dir / f"{ap.stem}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()