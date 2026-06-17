from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydub import AudioSegment


@dataclass
class BoundaryConfig:
    frame_ms: int = 10
    pre_roll_ms: int = 25
    post_roll_ms: int = 160
    final_post_roll_ms: int = 220
    search_before_ms: int = 160
    search_after_ms: int = 900
    min_clip_ms: int = 70
    min_gap_ms: int = 130
    onset_score: float = 0.55
    release_score: float = 0.32
    min_onset_ms: int = 20
    min_release_ms: int = 120


@dataclass
class BoundaryAnalysis:
    frame_times: np.ndarray
    frame_db: np.ndarray
    speech_score: np.ndarray
    noise_floor_db: float
    speech_peak_db: float
    onset_threshold_db: float
    release_threshold_db: float
    vad_available: bool
    vad_source: str


def _quality_for_phrase(start: float, end: float, duration: float) -> str:
    span = max(0, end - start)
    if span < 0.12 or start < 0 or end > duration + 0.05:
        return "bad"
    if span < 0.24:
        return "warn"
    return "good"


def _audio_samples(audio_path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_file(audio_path).set_channels(1)
    if sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    max_value = float(1 << (8 * audio.sample_width - 1))
    if max_value > 0:
        samples = samples / max_value
    return samples, int(audio.frame_rate)


def _frame_features(samples: np.ndarray, sample_rate: int, frame_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    if len(samples) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty
    frame_count = max(1, int(np.ceil(len(samples) / frame_len)))
    db = np.empty(frame_count, dtype=np.float32)
    times = np.empty(frame_count, dtype=np.float32)
    flux = np.zeros(frame_count, dtype=np.float32)
    previous_spectrum: np.ndarray | None = None
    for index in range(frame_count):
        start = index * frame_len
        end = min(len(samples), start + frame_len)
        frame = samples[start:end]
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        db[index] = 20 * np.log10(max(rms, 1e-6))
        times[index] = (start + end) / 2 / sample_rate

        windowed = frame * np.hanning(len(frame)) if len(frame) > 1 else frame
        spectrum = np.abs(np.fft.rfft(windowed))
        if previous_spectrum is not None and len(spectrum) == len(previous_spectrum):
            flux[index] = float(np.mean(np.maximum(spectrum - previous_spectrum, 0)))
        previous_spectrum = spectrum

    if np.max(flux) > 0:
        flux = flux / np.max(flux)
    return times, db, flux


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1 / (1 + np.exp(-values))


def _silero_intervals(audio_path: Path, duration: float) -> tuple[list[tuple[float, float]], str]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

        model = load_silero_vad(onnx=True)
        wav = read_audio(str(audio_path), sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
        intervals = [(float(item["start"]), float(item["end"])) for item in timestamps]
        return intervals, "silero-vad"
    except Exception:
        pass

    try:
        import torch

        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
        get_speech_timestamps, _, read_audio, _, _ = utils
        wav = read_audio(str(audio_path), sampling_rate=16000)
        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
        intervals = [(float(item["start"]), float(item["end"])) for item in timestamps]
        return intervals, "silero-vad-torchhub"
    except Exception:
        return [(0.0, duration)] if duration > 0 else [], "energy-only"


def analyze_audio(audio_path: Path, duration: float, config: BoundaryConfig | None = None) -> BoundaryAnalysis:
    config = config or BoundaryConfig()
    samples, sample_rate = _audio_samples(audio_path)
    frame_times, frame_db, flux = _frame_features(samples, sample_rate, config.frame_ms)
    if len(frame_times) == 0:
        empty = np.array([], dtype=np.float32)
        return BoundaryAnalysis(empty, empty, empty, -60.0, -60.0, -45.0, -52.0, False, "empty")

    noise_floor = float(np.percentile(frame_db, 20))
    speech_peak = float(np.percentile(frame_db, 95))
    dynamic_range = max(1.0, speech_peak - noise_floor)
    onset_threshold = noise_floor + min(16.0, max(8.0, dynamic_range * 0.35))
    release_threshold = noise_floor + min(9.0, max(4.0, dynamic_range * 0.18))

    energy_score = _sigmoid((frame_db - release_threshold) / 4.0)
    vad_score = np.zeros_like(energy_score)
    intervals, vad_source = _silero_intervals(audio_path, duration)
    vad_available = vad_source != "energy-only"
    for start, end in intervals:
        vad_score[(frame_times >= max(0.0, start - 0.03)) & (frame_times <= min(duration, end + 0.03))] = 1.0

    if vad_available:
        speech_score = 0.6 * energy_score + 0.25 * vad_score + 0.15 * flux
    else:
        speech_score = 0.85 * energy_score + 0.15 * flux

    return BoundaryAnalysis(
        frame_times=frame_times,
        frame_db=frame_db,
        speech_score=np.clip(speech_score, 0.0, 1.0),
        noise_floor_db=round(noise_floor, 2),
        speech_peak_db=round(speech_peak, 2),
        onset_threshold_db=round(onset_threshold, 2),
        release_threshold_db=round(release_threshold, 2),
        vad_available=vad_available,
        vad_source=vad_source,
    )


def _clip_indices(phrases: list[dict[str, Any]]) -> list[int]:
    return [index for index, phrase in enumerate(phrases) if phrase.get("kind") == "clip"]


def _neighbor_clip_bounds(phrases: list[dict[str, Any]], index: int) -> tuple[float | None, float | None]:
    previous_end: float | None = None
    next_start: float | None = None
    for candidate in reversed(phrases[:index]):
        if candidate.get("kind") == "clip":
            previous_end = float(candidate.get("end", 0))
            break
    for candidate in phrases[index + 1 :]:
        if candidate.get("kind") == "clip":
            next_start = float(candidate.get("start", 0))
            break
    return previous_end, next_start


def _hysteresis_segments(
    analysis: BoundaryAnalysis,
    left: float,
    right: float,
    config: BoundaryConfig,
    post_roll_ms: int,
) -> list[dict[str, float]]:
    times = analysis.frame_times
    scores = analysis.speech_score
    if len(times) == 0:
        return []
    in_window = np.where((times >= left) & (times <= right))[0]
    if len(in_window) == 0:
        return []

    min_onset_frames = max(1, int(round(config.min_onset_ms / config.frame_ms)))
    min_release_frames = max(1, int(round(config.min_release_ms / config.frame_ms)))
    min_clip_s = config.min_clip_ms / 1000
    segments: list[dict[str, float]] = []
    active = False
    start_time = left
    onset_count = 0
    release_count = 0

    for frame_index in in_window:
        score = float(scores[frame_index])
        frame_time = float(times[frame_index])
        if not active:
            if score >= config.onset_score:
                onset_count += 1
                if onset_count >= min_onset_frames:
                    active = True
                    start_idx = max(0, frame_index - onset_count + 1)
                    start_time = max(left, float(times[start_idx]) - config.pre_roll_ms / 1000)
                    release_count = 0
            else:
                onset_count = 0
            continue

        if score < config.release_score:
            release_count += 1
            if release_count >= min_release_frames:
                end_idx = max(0, frame_index - release_count + 1)
                end_time = min(right, float(times[end_idx]) + post_roll_ms / 1000)
                if end_time - start_time >= min_clip_s:
                    segments.append({"start": round(start_time, 3), "end": round(end_time, 3)})
                active = False
                onset_count = 0
                release_count = 0
        else:
            release_count = 0

    if active:
        end_time = min(right, float(times[in_window[-1]]) + post_roll_ms / 1000)
        if end_time - start_time >= min_clip_s:
            segments.append({"start": round(start_time, 3), "end": round(end_time, 3)})

    return _merge_close_segments(segments, config.min_gap_ms / 1000)


def _merge_close_segments(segments: list[dict[str, float]], min_gap_s: float) -> list[dict[str, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        if segment["start"] - merged[-1]["end"] < min_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(segment)
    return merged


def _removed_gaps(segments: list[dict[str, float]], min_gap_s: float) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for left, right in zip(segments, segments[1:]):
        if right["start"] - left["end"] >= min_gap_s:
            gaps.append({"start": left["end"], "end": right["start"], "reason": "internal-silence"})
    return gaps


def _lowest_energy_cut(analysis: BoundaryAnalysis, start: float, end: float) -> float:
    mask = np.where((analysis.frame_times >= start) & (analysis.frame_times <= end))[0]
    if len(mask) == 0:
        return round((start + end) / 2, 3)
    local = mask[int(np.argmin(analysis.frame_db[mask]))]
    return round(float(analysis.frame_times[local]), 3)


def _lowest_energy_cut_around(
    analysis: BoundaryAnalysis,
    boundary: float,
    left_limit: float,
    right_limit: float,
    radius_s: float = 0.08,
) -> float:
    start = max(left_limit, boundary - radius_s)
    end = min(right_limit, boundary + radius_s)
    if end <= start:
        return round(boundary, 3)
    return _lowest_energy_cut(analysis, start, end)


def _resolve_overlaps(phrases: list[dict[str, Any]], analysis: BoundaryAnalysis, duration: float) -> list[dict[str, Any]]:
    clip_positions = _clip_indices(phrases)
    previous_phrase: dict[str, Any] | None = None
    for position in clip_positions:
        phrase = phrases[position]
        segments = phrase.get("segments") or [{"start": phrase["start"], "end": phrase["end"]}]
        phrase["segments"] = [dict(segment) for segment in segments if segment["end"] > segment["start"]]
        if not phrase["segments"]:
            previous_phrase = phrase
            continue

        if previous_phrase and previous_phrase.get("segments"):
            previous_segment = previous_phrase["segments"][-1]
            current_segment = phrase["segments"][0]
            overlap_start = max(float(previous_segment["start"]), float(current_segment["start"]))
            overlap_end = min(float(previous_segment["end"]), float(current_segment["end"]))
            if overlap_end > overlap_start:
                cut = _lowest_energy_cut(analysis, overlap_start, overlap_end)
                previous_segment["end"] = max(float(previous_segment["start"]) + 0.001, cut)
                current_segment["start"] = min(float(current_segment["end"]) - 0.001, cut)
                previous_phrase.setdefault("boundaryCuts", []).append(
                    {"at": cut, "with": phrase.get("id"), "reason": "overlap-lowest-energy"}
                )
                phrase.setdefault("boundaryCuts", []).append(
                    {"at": cut, "with": previous_phrase.get("id"), "reason": "overlap-lowest-energy"}
                )
            else:
                gap = float(current_segment["start"]) - float(previous_segment["end"])
                if 0 <= gap <= 0.04:
                    boundary = (float(previous_segment["end"]) + float(current_segment["start"])) / 2
                    cut = _lowest_energy_cut_around(
                        analysis,
                        boundary,
                        float(previous_segment["start"]),
                        float(current_segment["end"]),
                    )
                    previous_segment["end"] = max(float(previous_segment["start"]) + 0.001, cut)
                    current_segment["start"] = min(float(current_segment["end"]) - 0.001, cut)
                    previous_phrase.setdefault("boundaryCuts", []).append(
                        {"at": cut, "with": phrase.get("id"), "reason": "adjacent-lowest-energy"}
                    )
                    phrase.setdefault("boundaryCuts", []).append(
                        {"at": cut, "with": previous_phrase.get("id"), "reason": "adjacent-lowest-energy"}
                    )

        phrase["segments"] = [
            {"start": round(max(0.0, float(segment["start"])), 3), "end": round(min(duration, float(segment["end"])), 3)}
            for segment in phrase["segments"]
            if float(segment["end"]) - float(segment["start"]) >= 0.03
        ]
        if phrase["segments"]:
            phrase["start"] = phrase["segments"][0]["start"]
            phrase["end"] = phrase["segments"][-1]["end"]
            phrase["removedGaps"] = _removed_gaps(phrase["segments"], 0.001)
            if not phrase["removedGaps"]:
                phrase.pop("removedGaps", None)
            phrase["quality"] = _quality_for_phrase(float(phrase["start"]), float(phrase["end"]), duration)
        previous_phrase = phrase

    for phrase in phrases:
        if phrase.get("kind") != "clip" or not phrase.get("segments"):
            continue
        phrase["segments"] = [
            {"start": round(max(0.0, float(segment["start"])), 3), "end": round(min(duration, float(segment["end"])), 3)}
            for segment in phrase["segments"]
            if float(segment["end"]) - float(segment["start"]) >= 0.03
        ]
        if not phrase["segments"]:
            continue
        phrase["start"] = phrase["segments"][0]["start"]
        phrase["end"] = phrase["segments"][-1]["end"]
        phrase["removedGaps"] = _removed_gaps(phrase["segments"], 0.001)
        if not phrase["removedGaps"]:
            phrase.pop("removedGaps", None)
        phrase["quality"] = _quality_for_phrase(float(phrase["start"]), float(phrase["end"]), duration)
    return phrases


def refine_phrase_boundaries_professional(
    audio_path: Path,
    phrases: list[dict[str, Any]],
    duration: float,
    config: BoundaryConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or BoundaryConfig()
    analysis = analyze_audio(audio_path, duration, config)
    if len(analysis.frame_times) == 0:
        return phrases

    refined: list[dict[str, Any]] = []
    clip_positions = _clip_indices(phrases)
    final_clip_position = clip_positions[-1] if clip_positions else -1

    for index, phrase in enumerate(phrases):
        if phrase.get("kind") != "clip":
            pause = {**phrase}
            pause["segments"] = []
            pause["ownership"] = "none"
            refined.append(pause)
            continue

        raw_start = float(phrase["start"])
        raw_end = float(phrase["end"])
        previous_end, next_start = _neighbor_clip_bounds(phrases, index)
        left = max(0.0, raw_start - config.search_before_ms / 1000)
        right = min(duration, raw_end + config.search_after_ms / 1000)
        if previous_end is not None:
            left = max(left, (previous_end + raw_start) / 2)
        if next_start is not None:
            right = min(right, (raw_end + next_start) / 2)

        post_roll = config.final_post_roll_ms if index == final_clip_position else config.post_roll_ms
        segments = _hysteresis_segments(analysis, left, right, config, post_roll)
        if not segments:
            segments = [{"start": round(max(0.0, raw_start - config.pre_roll_ms / 1000), 3), "end": round(min(duration, raw_end + post_roll / 1000), 3)}]

        updated = {**phrase}
        updated["anchor"] = {"start": round(raw_start, 3), "end": round(raw_end, 3), "source": phrase.get("source", "whisperx")}
        updated["segments"] = segments
        updated["start"] = segments[0]["start"]
        updated["end"] = segments[-1]["end"]
        updated["removedGaps"] = _removed_gaps(segments, config.min_gap_ms / 1000)
        if not updated["removedGaps"]:
            updated.pop("removedGaps", None)
        updated["scores"] = {
            "noiseFloorDb": analysis.noise_floor_db,
            "speechPeakDb": analysis.speech_peak_db,
            "onsetThresholdDb": analysis.onset_threshold_db,
            "releaseThresholdDb": analysis.release_threshold_db,
            "vadSource": analysis.vad_source,
            "vadAvailable": analysis.vad_available,
        }
        updated["ownership"] = "exclusive"
        updated["quality"] = _quality_for_phrase(updated["start"], updated["end"], duration)
        updated["source"] = "boundary-refined"
        refined.append(updated)

    return _resolve_overlaps(refined, analysis, duration)
