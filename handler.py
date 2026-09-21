"""RunPod serverless handler for VibeVoice-Realtime (Microsoft, MIT).

Built against the REAL upstream API, verified against
microsoft/VibeVoice at main on 2026-09-21:

    vibevoice.modular.modeling_vibevoice_streaming_inference
        .VibeVoiceStreamingForConditionalGenerationInference
    vibevoice.processor.vibevoice_streaming_processor
        .VibeVoiceStreamingProcessor

Do NOT reintroduce `from vibevoice.inference import load_model, tts_generate`.
That module has never existed. A published RunPod Hub listing
(jords1755/VibeVoice) imports it and dies at container start.
"""

import base64
import copy
import io
import os
import time
import traceback
from pathlib import Path

import soundfile as sf
import torch
import runpod
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# VibeVoice emits 24 kHz audio. The broken Hub listing wrote 16 kHz,
# which would have pitched every voice down by a third.
SAMPLE_RATE = 24000

MODEL_PATH = os.getenv("MODEL_PATH", "/models/vibevoice-realtime-0.5b")
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/models/voices"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "en-Emma_woman")
DEFAULT_CFG_SCALE = float(os.getenv("CFG_SCALE", "1.5"))
DDPM_STEPS = int(os.getenv("DDPM_STEPS", "5"))

# flash_attention_2 is NOT a dependency of the upstream package and needs
# nvcc to build. This image ships sdpa. Upstream states that only
# flash_attention_2 is fully tested, and that sdpa "may result in lower
# audio quality". Set ATTN_IMPL to change it, once a wheel is present.
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"[vibevoice] device={DEVICE} dtype={DTYPE} attn={ATTN_IMPL}")
print(f"[vibevoice] loading processor from {MODEL_PATH}")
PROCESSOR = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)

print(f"[vibevoice] loading model from {MODEL_PATH}")
try:
    MODEL = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation=ATTN_IMPL,
    )
except Exception as exc:  # noqa: BLE001
    if ATTN_IMPL == "flash_attention_2":
        print(f"[vibevoice] flash_attention_2 unavailable: {exc}. Falling back to sdpa.")
        MODEL = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="sdpa",
        )
    else:
        raise

MODEL.eval()
MODEL.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
print(f"[vibevoice] model ready, ddpm_steps={DDPM_STEPS}")


def _load_presets() -> dict:
    """Load every .pt voice preset once. These masters are never mutated."""
    presets = {}
    if not VOICES_DIR.exists():
        print(f"[vibevoice] WARNING: voices dir missing at {VOICES_DIR}")
        return presets
    target = DEVICE if DEVICE != "cpu" else "cpu"
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        for pt_file in sorted(VOICES_DIR.rglob("*.pt")):
            name = pt_file.stem
            presets[name] = torch.load(pt_file, map_location=target, weights_only=True)
    print(f"[vibevoice] loaded {len(presets)} voices: {', '.join(sorted(presets))}")
    return presets


VOICES = _load_presets()


def _resolve_voice(requested: str) -> str:
    """Exact match, then case-insensitive match, then unique substring."""
    if requested in VOICES:
        return requested
    lowered = {name.lower(): name for name in VOICES}
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    hits = [name for key, name in lowered.items() if requested.lower() in key]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"Voice '{requested}' is ambiguous. It matches: {', '.join(sorted(hits))}."
        )
    raise ValueError(
        f"Unknown voice '{requested}'. Available: {', '.join(sorted(VOICES))}."
    )


def synthesize(text: str, voice: str, cfg_scale: float) -> dict:
    master = VOICES[voice]

    # The master preset must survive every request unchanged. Both the
    # processor and generate() receive their own deep copies. Upstream's
    # one-shot demo can be looser; a warm serverless worker cannot.
    inputs = PROCESSOR.process_input_with_cached_prompt(
        text=text,
        cached_prompt=copy.deepcopy(master),
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(DEVICE)

    started = time.time()
    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=PROCESSOR.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        all_prefilled_outputs=copy.deepcopy(master),
    )
    elapsed = time.time() - started

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("Model returned no audio.")

    audio = outputs.speech_outputs[0].detach().to(torch.float32).cpu().numpy()
    audio = audio.reshape(-1)

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    duration = len(audio) / SAMPLE_RATE
    return {
        "audio_base64": base64.b64encode(buffer.read()).decode("utf-8"),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": round(duration, 3),
        "generation_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / duration, 3) if duration > 0 else None,
        "voice": voice,
        "cfg_scale": cfg_scale,
    }


def handler(job):
    """Input:  {"text": str, "voice": str?, "cfg_scale": float?}
    Output: {"audio_base64": str, "sample_rate": 24000, ...}
    """
    job_input = job.get("input") or {}

    text = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "Field 'text' is required and must be a non-empty string."}
    text = text.strip().replace("’", "'").replace("“", '"').replace("”", '"')

    if not VOICES:
        return {"error": f"No voice presets found in {VOICES_DIR}."}

    try:
        voice = _resolve_voice(str(job_input.get("voice") or DEFAULT_VOICE))
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        cfg_scale = float(job_input.get("cfg_scale", DEFAULT_CFG_SCALE))
    except (TypeError, ValueError):
        return {"error": "Field 'cfg_scale' must be a number."}

    try:
        return synthesize(text, voice, cfg_scale)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise RuntimeError(f"Inference failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
