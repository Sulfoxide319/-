from __future__ import annotations

from pathlib import Path
from typing import Any

from pydub import AudioSegment, effects

from .storage import RENDERS_DIR, list_recordings, new_id


def _safe_ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _clip_quality(segment: AudioSegment) -> AudioSegment:
    if len(segment) == 0:
        return segment
    segment = segment.set_channels(1).set_frame_rate(44100)
    if segment.dBFS != float("-inf"):
        target = -18.0
        segment = segment.apply_gain(max(-8.0, min(8.0, target - segment.dBFS)))
    fade = min(8, max(3, len(segment) // 16))
    return segment.fade_in(fade).fade_out(fade)


def render_track(items: list[dict[str, Any]]) -> dict[str, Any]:
    recordings = {recording["id"]: recording for recording in list_recordings()}
    output = AudioSegment.silent(duration=0, frame_rate=44100).set_channels(1)
    default_crossfade_ms = 0

    for item in items:
        if item.get("type") == "pause":
            duration = int(item.get("durationMs") or 250)
            output += AudioSegment.silent(duration=max(40, duration), frame_rate=44100).set_channels(1)
            continue

        recording_id = item.get("recordingId")
        phrase_id = item.get("phraseId")
        recording = recordings.get(recording_id)
        if not recording:
            raise ValueError(f"找不到录音：{recording_id}")

        phrase = next((p for p in recording.get("phrases", []) if p.get("id") == phrase_id), None)
        if not phrase:
            raise ValueError(f"找不到短语：{phrase_id}")
        if phrase.get("kind") == "pause":
            output += AudioSegment.silent(duration=260, frame_rate=44100).set_channels(1)
            continue

        audio = AudioSegment.from_file(recording["audioPath"])
        margin_ms = int(item.get("marginMs") or 12)
        start_ms = max(0, _safe_ms(float(phrase["start"])) - margin_ms)
        end_ms = min(len(audio), _safe_ms(float(phrase["end"])) + margin_ms)
        segment = _clip_quality(audio[start_ms:end_ms])
        crossfade_ms = max(0, min(8, int(item.get("crossfadeMs") or default_crossfade_ms)))
        if len(output) and crossfade_ms and len(segment) > crossfade_ms and len(output) > crossfade_ms:
            output = output.append(segment, crossfade=crossfade_ms)
        else:
            output += segment
        pause_after_ms = int(item.get("pauseAfterMs") or phrase.get("pauseAfterMs") or 0)
        if pause_after_ms > 0:
            output += AudioSegment.silent(duration=min(1200, pause_after_ms), frame_rate=44100).set_channels(1)

    if len(output) == 0:
        output = AudioSegment.silent(duration=300, frame_rate=44100).set_channels(1)
    output = effects.normalize(output, headroom=1.0)
    render_id = new_id("render")
    wav_path = RENDERS_DIR / f"{render_id}.wav"
    mp3_path = RENDERS_DIR / f"{render_id}.mp3"
    output.export(wav_path, format="wav")
    output.export(mp3_path, format="mp3", bitrate="192k")
    return {
        "id": render_id,
        "durationMs": len(output),
        "wavUrl": f"/api/renders/{render_id}.wav",
        "mp3Url": f"/api/renders/{render_id}.mp3",
    }
