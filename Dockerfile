# VibeVoice-Realtime serverless worker for RunPod.
#
# Weights and voice presets are baked in at build time. A serverless TTS
# that downloads a model on cold start feels broken to the caller.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Torch first, so the CUDA build is fixed before anything can pull a
# different one as a transitive dependency.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchaudio==2.6.0

# Upstream pins transformers==4.51.3 for the streaming TTS path.
# This is an exact pin, not a floor. Do not relax it.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pinned to an exact commit. Microsoft rewrote this repository's history
# when they re-enabled it in 2025, so a floating main is not safe here.
ARG VIBEVOICE_SHA=1541f590c7099820f10ea012f48d2399282df69f
RUN git clone https://github.com/microsoft/VibeVoice.git /opt/VibeVoice \
    && cd /opt/VibeVoice \
    && git checkout ${VIBEVOICE_SHA} \
    && pip3 install --no-cache-dir --no-deps .

# Bake the weights and the voice presets.
COPY builder/download_assets.py /app/builder/download_assets.py
RUN python3 /app/builder/download_assets.py

COPY handler.py .
COPY test_two_requests.py .

# Fail the build, not the first paying request, if the import graph is wrong.
RUN python3 -c "\
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference; \
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor; \
print('import graph OK')"

CMD ["python3", "-u", "handler.py"]
