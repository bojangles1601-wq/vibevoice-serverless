# Live probe results — 2026-09-21

Run inside the built image on an RTX 4070, CUDA 12.4, `ATTN_IMPL=sdpa`.
**Not a simulation and not a syntax check.** The model loaded, the voice preset
loaded, and audio came out.

## Build

All 12 stages passed, including the import-graph guard at stage 12:

```
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
print('import graph OK')
```

That guard exists because the published `jords1755/VibeVoice` Hub listing fails
exactly there, at container start, on an import of a module that does not exist.
This build fails at build time instead of on a paying request.

Image size: 7.61 GB. Weights and 26 voice presets are baked in.

## `test_two_requests.py` — PASSED

| Request | Audio produced | Sample rate | Real-time factor |
|---|---|---|---|
| First | 5.867 s | 24000 Hz | 0.296 |
| Second, same warm model | 4.933 s | 24000 Hz | 0.228 |

Empty text returned a clean error and did not crash:
`Field 'text' is required and must be a non-empty string.`

A real-time factor near 0.25 means the worker generates roughly four seconds of
speech per second of compute, on a mid-range consumer card.

## ⚠ The 16 percent gap, chased rather than accepted

The two requests used identical text, identical voice and greedy decoding, yet
the second was 16 percent shorter. That is the exact shape of the bug this
worker is built to prevent: a cached voice prompt degrading across a warm
worker's life. The probe's threshold passed it. **That is not the same as it
being fine**, so it was tested directly.

Five identical requests against one loaded model:

```
5.2, 5.467, 4.933, 4.933, 5.067  seconds
spread 9.8 percent
monotonically shrinking: False
```

**Verdict: variance, not decay.** Durations move both up and down and settle
around 5 seconds. A state-carryover defect would shrink monotonically. The first
request being longest is a cold-start artefact, not the start of a slide.


## ⭐ The real entry point, which the first probe never exercised

The first probe imported `handler` and called the function directly. That skips
the `if __name__ == "__main__"` guard, so **`runpod.serverless.start()` had never
run** — and that line is what the container's `CMD` actually invokes. Checking
the parts while skipping the entry point is the same mistake the broken listing
made. So it was run properly:

```
docker run --rm --gpus all vibevoice-serverless:dev \
    python3 -u handler.py --test_input '{"input":{"text":"Hello. This worker is online and speaking."}}'
```

Result, from the RunPod SDK's own log lines:

```
[vibevoice] model ready, ddpm_steps=5
[vibevoice] loaded 25 voices
--- Starting Serverless Worker |  Version 1.12.0 ---
INFO | Job local_test completed successfully.
```

Returned payload: 3.867 s of audio, 24000 Hz, real-time factor 0.309.

**Exit code 0. Total wall clock, container start to exit: 12 seconds.**

That 12 seconds is the cold start with weights baked in. Every test in
`tests.json` allows 180 seconds, so there is a 15x margin for a RunPod host
slower than this one.

⚠ **One count corrected.** The image loads **25** voice presets, not 26. The
earlier figure was counted from a file listing that included a `.wav` demo file.
The running container reports 25, and the running container is the authority.

## What this does NOT establish

- **Audio quality was not judged.** Nobody has listened. Duration and sample
  rate are structural checks, not perceptual ones.
- **sdpa versus flash_attention_2 was not compared.** Upstream says only
  flash_attention_2 is fully tested and that sdpa may sound worse. That claim
  stands untested here.
- **RunPod's own environment was not used.** This ran on local Docker with the
  NVIDIA runtime. That is the same container, not the same host.
- **Long inputs were not tested.** The longest text here is one sentence.
- **GPU memory use was not measured.** It fits on a 12 GB card with the
  desktop running, but the exact figure is unknown.

## Falsifier

If a listener reports that sdpa output is audibly worse than flash_attention_2,
or if duration drifts monotonically over a longer warm run, say fifty requests,
then the shipped default is wrong and the image needs a flash-attn wheel.
