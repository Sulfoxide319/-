from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydub.generators import Sine

from apps.api.audio_render import render_track
from apps.api.storage import RECORDINGS_DIR, recording_meta_path, write_json
from apps.api.transcription import fallback_transcribe


def main() -> int:
    source = Path("data/cache/regression_phrase.wav")
    source.parent.mkdir(parents=True, exist_ok=True)
    audio = Sine(440).to_audio_segment(duration=1200)
    audio.export(source, format="wav")

    result = fallback_transcribe(source, "我是张三，他是李四。")
    phrase_texts = [phrase["text"] for phrase in result["phrases"]]
    assert "，" not in phrase_texts
    assert "。" not in phrase_texts
    assert any(phrase.get("pauseAfterMs") for phrase in result["phrases"]), result["phrases"]

    recording_id = "rec_regression_audio_flow"
    recording_path = RECORDINGS_DIR / f"{recording_id}.wav"
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(recording_path, format="wav")
    meta = {
        "id": recording_id,
        "name": "regression.wav",
        "audioPath": str(recording_path),
        "audioUrl": f"/api/recordings/{recording_id}/audio",
        **result,
    }
    write_json(recording_meta_path(recording_id), meta)

    clips = [phrase for phrase in result["phrases"] if phrase["kind"] == "clip"]
    rendered = render_track(
        [
            {"type": "clip", "recordingId": recording_id, "phraseId": clips[0]["id"]},
            {"type": "clip", "recordingId": recording_id, "phraseId": clips[1]["id"]},
        ]
    )
    assert rendered["durationMs"] > 0
    print("regression ok", {"phrases": phrase_texts, "render": rendered})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
