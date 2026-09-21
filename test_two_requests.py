"""Live probe: two requests in ONE process, off ONE loaded model.

This is the test the upstream demo cannot give you. The demo runs once and
exits, so it can never show that generate() left the cached voice prompt
usable for a second call. A warm serverless worker serves many requests
from one load. If generate() mutates the preset, request two is garbage
and every later request stays garbage.

Pass condition: both WAVs exist, both have sane duration, and their
durations are within 25 percent of each other for the same-length text.
"""

import base64
import io
import sys

import soundfile as sf

from handler import handler

TEXT_A = "The quick brown fox jumps over the lazy dog, again and again."
TEXT_B = "The quick brown fox jumps over the lazy dog, again and again."


def run(name, payload):
    result = handler({"input": payload})
    if "error" in result:
        print(f"  {name}: ERROR {result['error']}")
        return None
    audio_bytes = base64.b64decode(result["audio_base64"])
    data, rate = sf.read(io.BytesIO(audio_bytes))
    sf.write(f"/tmp/{name}.wav", data, rate)
    print(
        f"  {name}: {result['duration_seconds']}s of audio at {rate} Hz, "
        f"rtf={result['real_time_factor']}, voice={result['voice']}"
    )
    return result


print("probe 1 of 3: first request")
first = run("probe_first", {"text": TEXT_A})

print("probe 2 of 3: second request, same warm model")
second = run("probe_second", {"text": TEXT_B})

print("probe 3 of 3: empty text must be rejected, not crash")
empty = handler({"input": {"text": "   "}})
print(f"  empty: {empty}")

failures = []
if not first:
    failures.append("first request failed")
if not second:
    failures.append("second request failed")
if first and second:
    a, b = first["duration_seconds"], second["duration_seconds"]
    if a <= 0.5 or b <= 0.5:
        failures.append(f"implausible durations: {a}s and {b}s")
    elif abs(a - b) / max(a, b) > 0.25:
        failures.append(f"second request diverged: {a}s then {b}s")
if "error" not in empty:
    failures.append("empty text was not rejected")

if failures:
    print("\nPROBE FAILED:")
    for item in failures:
        print(f"  - {item}")
    sys.exit(1)

print("\nPROBE PASSED: warm worker survives a second request.")
