from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydantic import BaseModel

from .audio_render import render_track
from .storage import (
    RECORDINGS_DIR,
    RENDERS_DIR,
    clear_generated_data,
    ensure_dirs,
    list_recordings,
    new_id,
    read_json,
    recording_meta_path,
    write_json,
)
from .transcription import transcribe


app = FastAPI(title="语音活字印刷拼接原型")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PhrasePatch(BaseModel):
    text: str | None = None
    start: float | None = None
    end: float | None = None


class RenderRequest(BaseModel):
    items: list[dict[str, Any]]
    settings: dict[str, Any] | None = None


def normalize_upload_to_wav(source_path: Path, wav_path: Path) -> float:
    try:
        audio = AudioSegment.from_file(source_path)
    except CouldntDecodeError as exc:
        raise HTTPException(status_code=400, detail="录音文件无法解码，请确认麦克风有输入并重新录制。") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="录音文件保存失败，请重新录制。") from exc

    duration_ms = len(audio)
    if duration_ms < 120:
        raise HTTPException(status_code=400, detail="录音太短，请至少录制半秒以上。")

    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")
    return duration_ms / 1000


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        import torch
        import whisperx  # noqa: F401

        whisperx_available = True
        cuda_available = bool(torch.cuda.is_available())
        torch_version = torch.__version__
    except Exception:
        whisperx_available = False
        cuda_available = False
        torch_version = None
    return {
        "ok": True,
        "whisperxAvailable": whisperx_available,
        "cudaAvailable": cuda_available,
        "torchVersion": torch_version,
    }


@app.get("/api/recordings")
def recordings() -> list[dict[str, Any]]:
    return list_recordings()


@app.delete("/api/data")
def clear_data() -> dict[str, Any]:
    return {"ok": True, "deleted": clear_generated_data()}


@app.post("/api/recordings")
async def upload_recording(
    file: UploadFile = File(...),
    manualText: str | None = Form(default=None),
    autoTranscribe: bool = Form(default=False),
    whisperModel: str | None = Form(default=None),
    whisperPrompt: str | None = Form(default=None),
    enableTextPostprocess: bool = Form(default=False),
) -> dict[str, Any]:
    ensure_dirs()
    recording_id = new_id("rec")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    raw_path = RECORDINGS_DIR / f"{recording_id}_source{suffix}"
    audio_path = RECORDINGS_DIR / f"{recording_id}.wav"
    with raw_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    normalize_upload_to_wav(raw_path, audio_path)

    result = transcribe(
        audio_path,
        manualText,
        prefer_whisperx=autoTranscribe,
        whisper_model=whisperModel,
        whisper_prompt=whisperPrompt,
        enable_text_postprocess=enableTextPostprocess,
    )
    meta = {
        "id": recording_id,
        "name": file.filename or recording_id,
        "audioPath": str(audio_path),
        "audioUrl": f"/api/recordings/{recording_id}/audio",
        **result,
    }
    write_json(recording_meta_path(recording_id), meta)
    return meta


@app.post("/api/recordings/{recording_id}/transcribe")
def transcribe_recording(
    recording_id: str,
    manualText: str | None = Form(default=None),
    autoTranscribe: bool = Form(default=True),
    whisperModel: str | None = Form(default=None),
    whisperPrompt: str | None = Form(default=None),
    enableTextPostprocess: bool = Form(default=False),
) -> dict[str, Any]:
    meta_path = recording_meta_path(recording_id)
    meta = read_json(meta_path, None)
    if not meta:
        raise HTTPException(status_code=404, detail="录音不存在")
    result = transcribe(
        Path(meta["audioPath"]),
        manualText,
        prefer_whisperx=autoTranscribe,
        whisper_model=whisperModel,
        whisper_prompt=whisperPrompt,
        enable_text_postprocess=enableTextPostprocess,
    )
    meta.update(result)
    write_json(meta_path, meta)
    return meta


@app.patch("/api/recordings/{recording_id}/phrases/{phrase_id}")
def patch_phrase(recording_id: str, phrase_id: str, patch: PhrasePatch) -> dict[str, Any]:
    meta_path = recording_meta_path(recording_id)
    meta = read_json(meta_path, None)
    if not meta:
        raise HTTPException(status_code=404, detail="录音不存在")
    for phrase in meta.get("phrases", []):
        if phrase.get("id") == phrase_id:
            if patch.text is not None:
                phrase["text"] = patch.text
            if patch.start is not None:
                phrase["start"] = max(0, round(patch.start, 3))
            if patch.end is not None:
                phrase["end"] = max(0, round(patch.end, 3))
            write_json(meta_path, meta)
            return phrase
    raise HTTPException(status_code=404, detail="短语不存在")


@app.post("/api/render")
def render(request: RenderRequest) -> dict[str, Any]:
    try:
        return render_track(request.items, request.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/recordings/{recording_id}/audio")
def recording_audio(recording_id: str) -> FileResponse:
    meta = read_json(recording_meta_path(recording_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(meta["audioPath"])


@app.get("/api/renders/{filename}")
def render_audio(filename: str) -> FileResponse:
    path = RENDERS_DIR / filename
    if not path.exists() or path.suffix.lower() not in {".wav", ".mp3"}:
        raise HTTPException(status_code=404, detail="输出不存在")
    return FileResponse(path)
