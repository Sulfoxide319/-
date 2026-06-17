from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .boundary_refine import refine_phrase_boundaries_professional
from .storage import DATA_DIR, new_id


DEFAULT_WHISPER_PROMPT = "这是中文语音拼接测试，请准确识别人名、短语和标点。"
WHISPER_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}
ZH_ALIGN_MODEL_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
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


def is_punctuation_text(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and all(char in PUNCTUATION for char in cleaned)


def punctuation_phrase(text: str, at: float, source: str) -> dict[str, Any]:
    at = round(max(0.0, at), 3)
    return {
        "id": new_id("phr"),
        "text": text,
        "start": at,
        "end": at,
        "kind": "pause",
        "quality": "good",
        "source": source,
        "pauseAfterMs": punctuation_pause_ms(text[-1]),
    }


def _duration_seconds(audio_path: Path) -> float:
    audio = AudioSegment.from_file(audio_path)
    return max(len(audio) / 1000, 0.01)


def fallback_transcribe(
    audio_path: Path,
    manual_text: str | None = None,
    whisper_model: str | None = None,
    enable_text_postprocess: bool = False,
    enable_boundary_refine: bool = False,
    enable_phrase_merge: bool = False,
) -> dict[str, Any]:
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
        "whisperModel": (whisper_model or "").strip() or "medium",
        "enableTextPostprocess": enable_text_postprocess,
        "enableBoundaryRefine": enable_boundary_refine,
        "enablePhraseMerge": enable_phrase_merge,
        "textPostprocessMode": "fallback",
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
                "source": "whisperx-postprocess",
            }
        )
        buffer.clear()

    for word in words:
        if "start" not in word or "end" not in word:
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        if is_punctuation_text(text):
            flush()
            phrases.append(punctuation_phrase(text, float(word["start"]), "whisperx-punctuation"))
            continue
        buffer.append({"text": text, "word": text, "start": word["start"], "end": word["end"]})
        if len("".join(item["text"] for item in buffer)) >= 2:
            flush()
    flush()
    return phrases


def words_to_raw_phrases(words: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    for word in words:
        if "start" not in word or "end" not in word:
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        start = float(word["start"])
        end = float(word["end"])
        if is_punctuation_text(text):
            phrases.append(punctuation_phrase(text, start, "whisperx-punctuation"))
            continue
        phrases.append(
            {
                "id": new_id("phr"),
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "kind": "clip",
                "quality": quality_for_phrase(start, end, duration),
                "source": "whisperx-word",
            }
        )
    return phrases


def _audio_samples(audio_path: Path) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    max_value = float(1 << (8 * audio.sample_width - 1))
    if max_value > 0:
        samples = samples / max_value
    return samples, int(audio.frame_rate)


def _frame_db(samples: np.ndarray, sample_rate: int, frame_ms: int = 10) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    if len(samples) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    frame_count = max(1, int(np.ceil(len(samples) / frame_len)))
    db = np.empty(frame_count, dtype=np.float32)
    times = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * frame_len
        end = min(len(samples), start + frame_len)
        frame = samples[start:end]
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        db[index] = 20 * np.log10(max(rms, 1e-6))
        times[index] = (start + end) / 2 / sample_rate
    return times, db


def refine_phrase_boundaries(
    audio_path: Path,
    phrases: list[dict[str, Any]],
    duration: float,
    pre_pad_ms: int = 20,
    post_pad_ms: int = 140,
    search_before_ms: int = 120,
    search_after_ms: int = 900,
    min_clip_ms: int = 80,
    min_silence_gap_ms: int = 120,
) -> list[dict[str, Any]]:
    samples, sample_rate = _audio_samples(audio_path)
    frame_times, frame_db = _frame_db(samples, sample_rate)
    if len(frame_times) == 0:
        return phrases

    peak = float(np.max(frame_db))
    floor = float(np.percentile(frame_db, 20))
    onset_threshold = max(floor + 12.0, peak - 34.0, -46.0)
    release_threshold = max(floor + 6.0, peak - 46.0, -56.0)
    active = frame_db >= release_threshold
    refined: list[dict[str, Any]] = []
    min_gap_s = min_silence_gap_ms / 1000

    def previous_clip_end(current_index: int, fallback: float) -> float:
        for candidate in reversed(phrases[:current_index]):
            if candidate.get("kind") == "clip":
                return float(candidate.get("end", fallback))
        return fallback

    def next_clip_start(current_index: int, fallback: float) -> float:
        for candidate in phrases[current_index + 1 :]:
            if candidate.get("kind") == "clip":
                return float(candidate.get("start", fallback))
        return fallback

    for index, phrase in enumerate(phrases):
        if phrase.get("kind") != "clip":
            refined.append(phrase)
            continue

        raw_start = float(phrase["start"])
        raw_end = float(phrase["end"])
        left_limit = max(0.0, raw_start - search_before_ms / 1000)
        right_limit = min(duration, raw_end + search_after_ms / 1000)
        prev_end = previous_clip_end(index, raw_start)
        if prev_end != raw_start:
            left_limit = max(left_limit, (prev_end + raw_start) / 2)
        next_start = next_clip_start(index, raw_end)
        if next_start != raw_end:
            right_limit = min(right_limit, (raw_end + next_start) / 2)

        in_window = (frame_times >= left_limit) & (frame_times <= right_limit)
        indices = np.where(in_window & active)[0]
        segments: list[dict[str, float]] = []
        if len(indices):
            groups: list[list[int]] = [[int(indices[0])]]
            for frame_index in indices[1:]:
                frame_time = float(frame_times[int(frame_index)])
                previous_time = float(frame_times[groups[-1][-1]])
                if frame_time - previous_time <= min_gap_s:
                    groups[-1].append(int(frame_index))
                else:
                    groups.append([int(frame_index)])

            for group in groups:
                group_db = frame_db[group]
                group_start = float(frame_times[group[0]])
                group_end = float(frame_times[group[-1]])
                has_onset = bool(np.max(group_db) >= onset_threshold)
                long_enough = group_end - group_start >= min_clip_ms / 1000
                if not has_onset or not long_enough:
                    continue
                segment_start = max(left_limit, group_start - pre_pad_ms / 1000)
                segment_end = min(right_limit, group_end + post_pad_ms / 1000)
                if segment_end - segment_start >= min_clip_ms / 1000:
                    segments.append({"start": round(segment_start, 3), "end": round(segment_end, 3)})

        if segments:
            new_start = segments[0]["start"]
            new_end = segments[-1]["end"]
        else:
            new_start = raw_start
            new_end = min(raw_end + post_pad_ms / 1000, right_limit)
            segments = [{"start": round(new_start, 3), "end": round(new_end, 3)}]

        updated = {**phrase}
        updated["start"] = round(max(0.0, new_start), 3)
        updated["end"] = round(min(duration, max(new_start + 0.001, new_end)), 3)
        updated["segments"] = segments
        if len(segments) > 1:
            updated["removedGaps"] = [
                {"start": segments[i]["end"], "end": segments[i + 1]["start"], "reason": "internal-silence"}
                for i in range(len(segments) - 1)
                if segments[i + 1]["start"] - segments[i]["end"] >= min_gap_s
            ]
        updated["quality"] = quality_for_phrase(updated["start"], updated["end"], duration)
        updated["source"] = "boundary-refined"
        refined.append(updated)

    return refined


def _jieba_tokens(text: str) -> list[str]:
    try:
        import jieba

        return [token for token in jieba.cut(text, HMM=True) if token.strip()]
    except Exception:
        return []


def merge_phrases(
    phrases: list[dict[str, Any]],
    duration: float,
    merge_max_gap_ms: int = 80,
) -> list[dict[str, Any]]:
    def merge_clip_run(clip_phrases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not clip_phrases:
            return []
        full_text = "".join(str(phrase.get("text", "")) for phrase in clip_phrases)
        force_words = {"愤怒", "神能", "人民", "制作", "大家", "我是", "原神", "启动", "我靠", "哇靠", "卧槽", "有病", "病吧"}
        token_lengths: list[int] = []
        force_matched = False
        index = 0
        while index < len(clip_phrases):
            match = ""
            for word in sorted(force_words, key=len, reverse=True):
                if full_text.startswith(word, index):
                    match = word
                    break
            if match:
                token_lengths.append(len(match))
                index += len(match)
                force_matched = True
            else:
                token_lengths.append(1)
                index += 1

        if not force_matched:
            tokens = _jieba_tokens(full_text)
            token_lengths = [len(token) for token in tokens] if tokens else []

        if not token_lengths or sum(token_lengths) != len(clip_phrases):
            token_lengths = []
            index = 0
            while index < len(clip_phrases):
                pair = "".join(str(phrase.get("text", "")) for phrase in clip_phrases[index : index + 2])
                if pair in force_words:
                    token_lengths.append(2)
                    index += 2
                else:
                    token_lengths.append(1)
                    index += 1

        merged_run: list[dict[str, Any]] = []
        cursor = 0
        for token_len in token_lengths:
            group = clip_phrases[cursor : cursor + token_len]
            cursor += token_len
            if not group:
                continue
            can_merge = len(group) > 1
            for left, right in zip(group, group[1:]):
                gap_ms = (float(right["start"]) - float(left["end"])) * 1000
                if gap_ms > merge_max_gap_ms:
                    can_merge = False
                    break
            if not can_merge:
                merged_run.extend(group)
                continue

            start = float(group[0]["start"])
            end = float(group[-1]["end"])
            merged_run.append(
                {
                    "id": new_id("phr"),
                    "text": "".join(str(phrase.get("text", "")) for phrase in group),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "kind": "clip",
                    "quality": quality_for_phrase(start, end, duration),
                    "source": "phrase-merged",
                    "children": [phrase.get("id") for phrase in group],
                }
            )
        return merged_run

    merged: list[dict[str, Any]] = []
    clip_buffer: list[dict[str, Any]] = []
    for phrase in phrases:
        if phrase.get("kind") == "clip":
            clip_buffer.append(phrase)
            continue
        merged.extend(merge_clip_run(clip_buffer))
        clip_buffer.clear()
        merged.append(phrase)
    merged.extend(merge_clip_run(clip_buffer))
    return merged


def _hf_cache_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        DATA_DIR / "cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    # Keep common Windows cache locations available even when HOME differs.
    candidates.append(home / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    roots: list[Path] = []
    for root in candidates:
        resolved = root.expanduser()
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return roots


def _repo_cache_dir(repo_id: str, cache_root: Path) -> Path:
    return cache_root / f"models--{repo_id.replace('/', '--')}"


def _latest_hf_snapshot(repo_id: str) -> Path | None:
    for cache_root in _hf_cache_roots():
        repo_dir = _repo_cache_dir(repo_id, cache_root)
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        ref_path = repo_dir / "refs" / "main"
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            ref_snapshot = snapshots_dir / revision
            if ref_snapshot.exists():
                return ref_snapshot

        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return max(snapshots, key=lambda path: path.stat().st_mtime)
    return None


def _resolve_whisper_model(model_name: str) -> tuple[str, str, bool]:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path), f"local path {model_path}", True

    repo_id = WHISPER_MODEL_REPOS.get(model_name, model_name if "/" in model_name else "")
    if repo_id:
        snapshot = _latest_hf_snapshot(repo_id)
        if snapshot and (snapshot / "model.bin").exists():
            return str(snapshot), f"local cache {repo_id}", True

    return model_name, f"remote model {model_name}", False


def _resolve_zh_align_model() -> tuple[str | None, bool]:
    snapshot = _latest_hf_snapshot(ZH_ALIGN_MODEL_REPO)
    if snapshot and snapshot.exists():
        return str(snapshot), True
    return None, False


def whisperx_transcribe(
    audio_path: Path,
    whisper_model: str | None = None,
    whisper_prompt: str | None = None,
    enable_text_postprocess: bool = False,
    enable_boundary_refine: bool = False,
    enable_phrase_merge: bool = False,
) -> dict[str, Any] | None:
    try:
        import torch
        import whisperx
    except Exception:
        return None

    model_name = (whisper_model or "").strip() or "medium"
    prompt = (whisper_prompt or "").strip() or DEFAULT_WHISPER_PROMPT
    audio_path = audio_path.resolve()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_ref, model_source, model_is_local = _resolve_whisper_model(model_name)
        model = whisperx.load_model(
            model_ref,
            device,
            language="zh",
            compute_type=compute_type,
            asr_options={"initial_prompt": prompt},
            download_root=str(DATA_DIR / "cache" / "huggingface"),
            local_files_only=model_is_local,
        )
        result = model.transcribe(str(audio_path), batch_size=4, language="zh")
        align_ref, align_is_local = _resolve_zh_align_model()
        if align_ref:
            model_a, metadata = whisperx.load_align_model(
                language_code="zh",
                device=device,
                model_name=align_ref,
                model_cache_only=align_is_local,
            )
        else:
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
        raw_phrases = words_to_raw_phrases(words, duration)
        phrases = raw_phrases
        mode = "raw"
        if enable_phrase_merge:
            phrases = merge_phrases(phrases, duration)
            mode = "phrase-merged"
        if enable_boundary_refine:
            phrases = refine_phrase_boundaries_professional(audio_path, phrases, duration)
            mode = "phrase-merged+boundary-refined" if enable_phrase_merge else "boundary-refined"
        if enable_text_postprocess and not enable_phrase_merge and not enable_boundary_refine:
            phrases = words_to_phrases(words, duration)
            mode = "legacy-postprocess"

        return {
            "text": text,
            "language": "zh",
            "duration": round(duration, 3),
            "segments": aligned.get("segments", []),
            "words": words,
            "phrases": phrases,
            "phrasesRaw": raw_phrases,
            "engine": "whisperx",
            "engineNote": f"WhisperX {model_name} on {device}, compute_type={compute_type}, {model_source}",
            "whisperModel": model_name,
            "whisperPrompt": prompt,
            "enableTextPostprocess": enable_text_postprocess,
            "enableBoundaryRefine": enable_boundary_refine,
            "enablePhraseMerge": enable_phrase_merge,
            "textPostprocessMode": mode,
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
            "whisperModel": model_name,
            "whisperPrompt": prompt,
            "enableTextPostprocess": enable_text_postprocess,
            "enableBoundaryRefine": enable_boundary_refine,
            "enablePhraseMerge": enable_phrase_merge,
            "textPostprocessMode": "error",
        }


def transcribe(
    audio_path: Path,
    manual_text: str | None,
    prefer_whisperx: bool,
    whisper_model: str | None = None,
    whisper_prompt: str | None = None,
    enable_text_postprocess: bool = False,
    enable_boundary_refine: bool = False,
    enable_phrase_merge: bool = False,
) -> dict[str, Any]:
    if prefer_whisperx:
        result = whisperx_transcribe(
            audio_path,
            whisper_model,
            whisper_prompt,
            enable_text_postprocess,
            enable_boundary_refine,
            enable_phrase_merge,
        )
        if result and result.get("phrases"):
            return result
        fallback = fallback_transcribe(
            audio_path,
            manual_text,
            whisper_model,
            enable_text_postprocess,
            enable_boundary_refine,
            enable_phrase_merge,
        )
        if result and result.get("engineNote"):
            fallback["engineNote"] = result["engineNote"]
        if result and result.get("engine"):
            fallback["engine"] = result["engine"]
        return fallback
    return fallback_transcribe(audio_path, manual_text, whisper_model, enable_text_postprocess, enable_boundary_refine, enable_phrase_merge)
