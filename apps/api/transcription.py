from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .storage import new_id


PUNCTUATION = set("，。！？；：,.!?;:")
PUNCTUATION_PAUSES_MS = {
    "，": 180,
    ",": 180,
    "、": 140,
    "。": 360,
    ".": 320,
    "！": 360,
    "!": 320,
    "？": 360,
    "?": 320,
    "；": 260,
    ";": 240,
    "：": 220,
    ":": 200,
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def text_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^rec_[0-9a-f]+_", "", stem)
    stem = re.sub(r"[_-]+", "", stem)
    return normalize_text(stem)


def split_text_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    units: list[str] = []
    current = ""
    for char in text:
        if char in PUNCTUATION:
            if current:
                units.append(current)
                current = ""
            units.append(char)
            continue
        current += char
        if len(current) >= 2:
            units.append(current)
            current = ""
    if current:
        units.append(current)
    return units


def punctuation_pause_ms(mark: str) -> int:
    return PUNCTUATION_PAUSES_MS.get(mark, 180)


def _duration_seconds(audio_path: Path) -> float:
    audio = AudioSegment.from_file(audio_path)
    return max(len(audio) / 1000, 0.01)


def fallback_transcribe(audio_path: Path, manual_text: str | None = None) -> dict[str, Any]:
    text = normalize_text(manual_text or "") or text_from_filename(audio_path) or "未命名录音"
    units = split_text_units(text) or [text]
    duration = _duration_seconds(audio_path)
    speech_units = [unit for unit in units if unit not in PUNCTUATION]
    total_chars = sum(max(len(unit), 1) for unit in speech_units) or 1
    cursor = 0.0
    words: list[dict[str, Any]] = []
    phrases: list[dict[str, Any]] = []

    for unit in units:
        if unit in PUNCTUATION:
            if phrases:
                phrases[-1]["punctuation"] = unit
                phrases[-1]["pauseAfterMs"] = punctuation_pause_ms(unit)
            continue
        unit_duration = duration * (max(len(unit), 1) / total_chars)
        start = cursor
        end = min(duration, cursor + unit_duration)
        cursor = end
        word = {"text": unit, "start": round(start, 3), "end": round(end, 3)}
        words.append(word)
        phrases.append(
            {
                "id": new_id("phr"),
                "text": unit,
                "start": round(start, 3),
                "end": round(end, 3),
                "kind": "clip",
                "quality": quality_for_phrase(start, end, duration),
                "source": "fallback",
            }
        )

    return {
        "text": text,
        "language": "zh",
        "duration": round(duration, 3),
        "words": words,
        "phrases": phrases,
        "engine": "fallback",
        "engineNote": "未检测到 WhisperX 或未请求真实转写，使用手工文本/文件名按时长均分生成短语。",
    }


def quality_for_phrase(start: float, end: float, duration: float) -> str:
    span = max(0, end - start)
    if span < 0.12 or start < 0 or end > duration + 0.05:
        return "bad"
    if span < 0.24:
        return "warn"
    return "good"


def words_to_phrases(words: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        text = "".join(str(item.get("word") or item.get("text") or "").strip() for item in buffer)
        start = float(buffer[0]["start"])
        end = float(buffer[-1]["end"])
        phrases.append(
            {
                "id": new_id("phr"),
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "kind": "clip",
                "quality": quality_for_phrase(start, end, duration),
                "source": "whisperx",
            }
        )
        buffer.clear()

    for word in words:
        if "start" not in word or "end" not in word:
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        buffer.append({"text": text, "word": text, "start": word["start"], "end": word["end"]})
        if len("".join(item["text"] for item in buffer)) >= 2:
            flush()
    flush()
    return phrases


def whisperx_transcribe(audio_path: Path) -> dict[str, Any] | None:
    try:
        import torch
        import whisperx
    except Exception:
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = whisperx.load_model("medium", device, language="zh", compute_type=compute_type)
        result = model.transcribe(str(audio_path), batch_size=4, language="zh")
        model_a, metadata = whisperx.load_align_model(language_code="zh", device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, str(audio_path), device)
        duration = _duration_seconds(audio_path)
        words = [
            {
                "text": str(word.get("word", "")).strip(),
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
            }
            for word in aligned.get("word_segments", [])
            if "start" in word and "end" in word
        ]
        text = normalize_text("".join(segment.get("text", "") for segment in aligned.get("segments", [])))
        if not text or not words:
            raise RuntimeError("WhisperX did not return aligned words")
        return {
            "text": text,
            "language": "zh",
            "duration": round(duration, 3),
            "segments": aligned.get("segments", []),
            "words": words,
            "phrases": words_to_phrases(words, duration),
            "engine": "whisperx",
            "engineNote": f"WhisperX on {device}, compute_type={compute_type}",
        }
    except Exception as exc:
        duration = None
        try:
            duration = round(_duration_seconds(audio_path), 3)
        except (CouldntDecodeError, FileNotFoundError, OSError):
            pass
        return {
            "text": "",
            "language": "zh",
            "duration": duration or 0,
            "words": [],
            "phrases": [],
            "engine": "whisperx-error",
            "engineNote": f"WhisperX 转写失败，已退回手工文本/文件名模式：{type(exc).__name__}: {exc}",
        }


def transcribe(audio_path: Path, manual_text: str | None, prefer_whisperx: bool) -> dict[str, Any]:
    if prefer_whisperx:
        result = whisperx_transcribe(audio_path)
        if result and result.get("phrases"):
            return result
        fallback = fallback_transcribe(audio_path, manual_text)
        if result and result.get("engineNote"):
            fallback["engineNote"] = result["engineNote"]
        return fallback
    return fallback_transcribe(audio_path, manual_text)
