from __future__ import annotations

import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api.audio_render import render_track
from apps.api.boundary_refine import refine_phrase_boundaries_professional
from apps.api.storage import RECORDINGS_DIR, RENDERS_DIR, recording_meta_path, write_json
from apps.api.transcription import merge_phrases, punctuation_phrase


def assert_no_clip_overlap(phrases: list[dict]) -> None:
    clips = [phrase for phrase in phrases if phrase.get("kind") == "clip"]
    for left, right in zip(clips, clips[1:]):
        left_segments = left.get("segments") or []
        right_segments = right.get("segments") or []
        if left_segments and right_segments:
            assert left_segments[-1]["end"] <= right_segments[0]["start"] + 0.001, (left, right)


def assert_pause_owns_no_audio(phrases: list[dict]) -> None:
    pauses = [phrase for phrase in phrases if phrase.get("kind") == "pause"]
    assert pauses, phrases
    for pause in pauses:
        assert pause.get("ownership") == "none" or not pause.get("segments"), pause
        assert not pause.get("segments"), pause


def build_fixture(path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = (
        AudioSegment.silent(duration=900)
        + Sine(440).to_audio_segment(duration=520).apply_gain(-8)
        + AudioSegment.silent(duration=340)
        + Sine(660).to_audio_segment(duration=720).apply_gain(-8)
        + AudioSegment.silent(duration=800)
    ).set_channels(1).set_frame_rate(16000)
    audio.export(path, format="wav")
    return len(audio) / 1000


def synthetic_boundary_acceptance() -> list[dict]:
    path = ROOT / "data" / "cache" / "acceptance_boundary.wav"
    duration = build_fixture(path)

    phrases = [
        {"id": "phr_a", "text": "原", "start": 0.78, "end": 1.18, "kind": "clip", "quality": "good", "source": "test"},
        {"id": "phr_b", "text": "神", "start": 1.18, "end": 1.48, "kind": "clip", "quality": "good", "source": "test"},
        punctuation_phrase(",", 1.49, "test-punctuation"),
        {"id": "phr_c", "text": "启", "start": 1.55, "end": 2.04, "kind": "clip", "quality": "good", "source": "test"},
        {"id": "phr_d", "text": "动", "start": 2.04, "end": 2.35, "kind": "clip", "quality": "good", "source": "test"},
    ]
    merged = merge_phrases(phrases, duration)
    refined = refine_phrase_boundaries_professional(path, merged, duration)

    assert_pause_owns_no_audio(refined)
    assert_no_clip_overlap(refined)
    clips = [phrase for phrase in refined if phrase.get("kind") == "clip"]
    assert all(phrase.get("ownership") == "exclusive" for phrase in clips), refined
    assert all(phrase.get("scores", {}).get("noiseFloorDb") is not None for phrase in clips), refined
    assert all(phrase.get("segments") for phrase in clips), refined
    assert clips[0]["segments"][0]["start"] > 0.75, refined
    assert clips[-1]["segments"][-1]["end"] >= 2.25, refined

    return refined


def colloquial_merge_acceptance() -> None:
    phrases = [
        {"id": "phr_wo", "text": "\u6211", "start": 0.1, "end": 0.3, "kind": "clip", "quality": "good", "source": "test"},
        {"id": "phr_kao", "text": "\u9760", "start": 0.3, "end": 0.8, "kind": "clip", "quality": "good", "source": "test"},
        punctuation_phrase("!", 0.8, "test-punctuation"),
        {"id": "phr_wa", "text": "\u54c7", "start": 1.2, "end": 1.5, "kind": "clip", "quality": "good", "source": "test"},
        {"id": "phr_kao2", "text": "\u9760", "start": 1.5, "end": 2.0, "kind": "clip", "quality": "good", "source": "test"},
    ]
    merged = merge_phrases(phrases, 2.5)
    texts = [phrase["text"] for phrase in merged]
    assert "\u6211\u9760" in texts, texts
    assert "\u54c7\u9760" in texts, texts
    assert any(phrase["kind"] == "pause" for phrase in merged), merged


def render_acceptance(refined: list[dict]) -> dict:
    recording_id = "rec_acceptance_boundary"
    recording_path = RECORDINGS_DIR / f"{recording_id}.wav"
    source_path = ROOT / "data" / "cache" / "acceptance_boundary.wav"
    recording_path.write_bytes(source_path.read_bytes())

    meta = {
        "id": recording_id,
        "name": "acceptance-boundary.wav",
        "audioPath": str(recording_path),
        "audioUrl": f"/api/recordings/{recording_id}/audio",
        "text": "".join(phrase["text"] for phrase in refined),
        "duration": 3.28,
        "phrases": refined,
        "engine": "acceptance",
    }
    write_json(recording_meta_path(recording_id), meta)

    items = [
        {"type": "clip", "recordingId": recording_id, "phraseId": phrase["id"]}
        for phrase in refined
        if phrase.get("kind") == "clip"
    ]
    rendered = render_track(items, None)
    full_span_ms = sum(round((phrase["end"] - phrase["start"]) * 1000) for phrase in refined if phrase.get("kind") == "clip")
    assert rendered["durationMs"] <= full_span_ms, (rendered, full_span_ms)
    assert rendered["durationMs"] > 500, rendered
    return rendered


def cleanup_acceptance_artifacts(rendered: dict | None = None) -> None:
    recording_id = "rec_acceptance_boundary"
    for path in (
        RECORDINGS_DIR / f"{recording_id}.wav",
        recording_meta_path(recording_id),
    ):
        if path.exists():
            path.unlink()
    if rendered:
        for key in ("wavUrl", "mp3Url"):
            name = Path(str(rendered.get(key, ""))).name
            if name:
                path = RENDERS_DIR / name
                if path.exists():
                    path.unlink()


def main() -> int:
    rendered: dict | None = None
    try:
        colloquial_merge_acceptance()
        refined = synthetic_boundary_acceptance()
        rendered = render_acceptance(refined)
        print(
            "acceptance boundary ok",
            {
                "phrases": [
                    {
                        "text": phrase["text"],
                        "kind": phrase["kind"],
                        "segments": phrase.get("segments"),
                        "ownership": phrase.get("ownership"),
                        "scores": phrase.get("scores"),
                    }
                    for phrase in refined
                ],
                "render": rendered,
            },
        )
    finally:
        cleanup_acceptance_artifacts(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
