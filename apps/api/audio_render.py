from __future__ import annotations

from typing import Any

from pydub import AudioSegment, effects

from .storage import RENDERS_DIR, list_recordings, new_id


DEFAULT_RENDER_SETTINGS: dict[str, Any] = {
    "enableAudioPostprocess": False,
    "marginMs": 0,
    "enableGainNormalize": False,
    "targetDbfs": -18,
    "maxGainDb": 8,
    "enableClipFade": False,
    "fadeMs": 0,
    "enableCrossfade": False,
    "crossfadeMs": 0,
    "enableFinalNormalize": False,
}


def _safe_ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_RENDER_SETTINGS, **(settings or {})}


def _format_segment(segment: AudioSegment) -> AudioSegment:
    return segment.set_channels(1).set_frame_rate(44100)


def _postprocess_clip(segment: AudioSegment, settings: dict[str, Any]) -> AudioSegment:
    segment = _format_segment(segment)
    if len(segment) == 0:
        return segment

    if settings["enableGainNormalize"] and segment.dBFS != float("-inf"):
        target = float(settings["targetDbfs"])
        max_gain = abs(float(settings["maxGainDb"]))
        segment = segment.apply_gain(max(-max_gain, min(max_gain, target - segment.dBFS)))

    if settings["enableClipFade"]:
        fade_ms = max(0, int(settings["fadeMs"]))
        if fade_ms:
            fade_ms = min(fade_ms, len(segment) // 2)
            segment = segment.fade_in(fade_ms).fade_out(fade_ms)

    return segment


def _phrase_ranges_ms(phrase: dict[str, Any], audio_length_ms: int, margin_ms: int) -> list[tuple[int, int]]:
    raw_segments = phrase.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        ranges = []
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            start = segment.get("start")
            end = segment.get("end")
            if start is None or end is None:
                continue
            start_ms = max(0, _safe_ms(float(start)) - margin_ms)
            end_ms = min(audio_length_ms, _safe_ms(float(end)) + margin_ms)
            if end_ms > start_ms:
                ranges.append((start_ms, end_ms))
        if ranges:
            return ranges

    start_ms = max(0, _safe_ms(float(phrase["start"])) - margin_ms)
    end_ms = min(audio_length_ms, _safe_ms(float(phrase["end"])) + margin_ms)
    return [(start_ms, end_ms)] if end_ms > start_ms else []


def render_track(items: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    render_settings = _settings(settings)
    enable_postprocess = bool(render_settings["enableAudioPostprocess"])
    recordings = {recording["id"]: recording for recording in list_recordings()}
    output = AudioSegment.silent(duration=0, frame_rate=44100).set_channels(1)

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
            duration = int(item.get("durationMs") or phrase.get("pauseAfterMs") or 260)
            output += AudioSegment.silent(duration=max(40, duration), frame_rate=44100).set_channels(1)
            continue

        audio = AudioSegment.from_file(recording["audioPath"])
        margin_ms = int(render_settings["marginMs"]) if enable_postprocess else 0
        ranges = _phrase_ranges_ms(phrase, len(audio), margin_ms)
        segment = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
        for start_ms, end_ms in ranges:
            segment += audio[start_ms:end_ms]
        segment = _postprocess_clip(segment, render_settings) if enable_postprocess else _format_segment(segment)

        crossfade_ms = 0
        if enable_postprocess and render_settings["enableCrossfade"]:
            crossfade_ms = max(0, int(render_settings["crossfadeMs"]))
            crossfade_ms = min(crossfade_ms, len(segment) // 2, len(output) // 2)

        if crossfade_ms:
            output = output.append(segment, crossfade=crossfade_ms)
        else:
            output += segment

        pause_after_ms = int(item.get("pauseAfterMs") or phrase.get("pauseAfterMs") or 0)
        if pause_after_ms > 0:
            output += AudioSegment.silent(duration=min(1200, pause_after_ms), frame_rate=44100).set_channels(1)

    if len(output) == 0:
        output = AudioSegment.silent(duration=300, frame_rate=44100).set_channels(1)

    if enable_postprocess and render_settings["enableFinalNormalize"]:
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
        "settingsUsed": render_settings,
    }
