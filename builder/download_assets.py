"""Bake model weights and voice presets into the image at build time."""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
MODEL_DIR = Path("/models/vibevoice-realtime-0.5b")
VOICES_DIR = Path("/models/voices")
UPSTREAM_VOICES = Path("/opt/VibeVoice/demo/voices/streaming_model")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_ID} ...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(MODEL_DIR),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)

if not UPSTREAM_VOICES.exists():
    raise SystemExit(f"Voice presets not found at {UPSTREAM_VOICES}")

count = 0
for pt_file in sorted(UPSTREAM_VOICES.rglob("*.pt")):
    shutil.copy2(pt_file, VOICES_DIR / pt_file.name)
    count += 1

if count == 0:
    raise SystemExit("No voice presets were copied. Refusing to build a mute image.")

print(f"Baked {count} voice presets into {VOICES_DIR}")
