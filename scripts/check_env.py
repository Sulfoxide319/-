from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    payload: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "inside_project_venv": ".venv" in Path(sys.executable).parts,
        "whisperx_available": module_available("whisperx"),
        "faster_whisper_available": module_available("faster_whisper"),
    }

    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        payload.update({"torch_error": repr(exc), "cuda_available": False})

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("inside_project_venv") and payload.get("whisperx_available") and payload.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
