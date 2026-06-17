from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
RENDERS_DIR = DATA_DIR / "renders"


def ensure_dirs() -> None:
    for path in (RECORDINGS_DIR, TRANSCRIPTIONS_DIR, RENDERS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def recording_meta_path(recording_id: str) -> Path:
    return TRANSCRIPTIONS_DIR / f"{recording_id}.json"


def list_recordings() -> list[dict[str, Any]]:
    ensure_dirs()
    recordings: list[dict[str, Any]] = []
    for path in sorted(TRANSCRIPTIONS_DIR.glob("rec_*.json")):
        recordings.append(read_json(path, {}))
    return recordings

