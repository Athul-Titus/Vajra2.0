"""Whisper + pyannote diarization (single file or folder).

Examples:
  set HF_TOKEN=***
  python transcribe_whisper.py --audio "audio transcript/athul1.ogg" --model base --out-dir "audio transcript"
  python transcribe_whisper.py --folder "audio transcript" --model base --out-dir "audio transcript"
"""
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper + diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", help="Path to one audio file")
    group.add_argument("--folder", help="Transcribe all audio files in a folder")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--language", default=None, help="Language hint (e.g., en, hi, ml); leave empty to auto-detect")
    parser.add_argument("--stereo-split", action="store_true", help="If stereo, left=User1 right=User2 (skip diarization)")
    parser.add_argument("--min-seg-dur", type=float, default=0.8, help="Min diarized segment length (s)")
    parser.add_argument("--merge-gap", type=float, default=0.3, help="Merge same-speaker if gap <= this (s)")
    parser.add_argument("--out-dir", default="audio transcript", help="Output directory for .txt files")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), help="HF token for pyannote")
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


def transcribe_one(audio_path: Path, wmodel, pipeline: Pipeline, device: torch.device, args: argparse.Namespace):
    wav, sr = load_audio(audio_path)
    segments = []
    mode = "diar"

    if args.stereo_split and wav.shape[0] == 2:
        duration = wav.shape[1] / sr
        segments.append((0.0, duration, "User1", wav[0]))
        segments.append((0.0, duration, "User2", wav[1]))
        mode = "stereo"
    else:
        segments = diarize_segments(wav, sr, pipeline, args)

    mono = wav.mean(dim=0)
    results = []
    for start, end, spk, pre in segments:
        if mode == "stereo" and pre is not None:
            audio_t = pre
        else:
            s_i = int(start * sr)
            e_i = int(end * sr)
            audio_t = mono[s_i:e_i]
        if audio_t.numel() == 0:
            continue
        audio_np = audio_t.numpy().astype(np.float32)
        text = wmodel.transcribe(audio_np, language=args.language, fp16=(device.type == "cuda")).get("text", "").strip()
        results.append((start, end, spk, text))

    speaker_map = {}
    user_counter = 1
    if mode != "stereo":
        for _, _, spk, _ in results:
            if spk not in speaker_map:
                speaker_map[spk] = f"User{user_counter}"
                user_counter += 1

    lines = []
    for start, end, spk, text in results:
        label = spk if mode == "stereo" else speaker_map.get(spk, spk)
        ts = f"[{start:6.2f}-{end:6.2f}]"
        lines.append(f"{ts} {label}: {text}")
    return lines


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("HF_TOKEN required for pyannote (set env or --hf-token)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wmodel = whisper.load_model(args.model, device=device.type)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token)
    pipeline.to(device)

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
            lines = transcribe_one(ap, wmodel, pipeline, device, args)
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