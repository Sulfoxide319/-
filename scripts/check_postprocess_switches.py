from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.audio_render import DEFAULT_RENDER_SETTINGS
from apps.api.transcription import words_to_phrases, words_to_raw_phrases


def main() -> None:
    words = [
        {"text": "人", "start": 0.0, "end": 0.1},
        {"text": "民", "start": 0.1, "end": 0.2},
        {"text": "制", "start": 0.2, "end": 0.3},
    ]
    raw = words_to_raw_phrases(words, duration=1.0)
    post = words_to_phrases(words, duration=1.0)

    assert [phrase["text"] for phrase in raw] == ["人", "民", "制"]
    assert all(phrase["source"] == "whisperx-word" for phrase in raw)
    assert len(raw) == len(words)

    assert [phrase["text"] for phrase in post] == ["人民", "制"]
    assert all(phrase["source"] == "whisperx-postprocess" for phrase in post)

    assert DEFAULT_RENDER_SETTINGS["enableAudioPostprocess"] is False
    assert DEFAULT_RENDER_SETTINGS["marginMs"] == 0
    assert DEFAULT_RENDER_SETTINGS["enableGainNormalize"] is False
    assert DEFAULT_RENDER_SETTINGS["enableClipFade"] is False
    assert DEFAULT_RENDER_SETTINGS["enableCrossfade"] is False
    assert DEFAULT_RENDER_SETTINGS["enableFinalNormalize"] is False



if __name__ == "__main__":
    main()
