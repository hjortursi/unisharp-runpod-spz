# UniSHARP RunPod serverless image: pre-baked deps, model, UniSHARP, UniK3D, Niantic SPZ.
# Build remotely (GitHub Actions), not on the Mac mini, to avoid local disk pressure.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UNISHARP_CHECKPOINT=/workspace/models/pretained_model.pt \
    UNISHARP_DEFAULT_CAMERA=auto \
    COMPACT_OUTPUT_DEFAULT=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      git git-lfs curl wget ca-certificates ffmpeg \
      build-essential ninja-build cmake python3-dev \
      libgl1 libglib2.0-0 libgomp1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && (git lfs install --skip-repo || true)

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir runpod requests huggingface_hub hf_transfer boto3

RUN git clone --depth 1 https://github.com/Insta360-Research-Team/UniSHARP.git /workspace/UniSHARP \
    && git clone --depth 1 https://github.com/lpiccinelli-eth/UniK3D.git /workspace/UniSHARP/UniK3D

WORKDIR /workspace/UniSHARP

# The base image is already torch 2.8 / CUDA 12.8. Do not reinstall torch: it is slow
# and bloats the remote build. Install UniSHARP deps with torch packages filtered out.
RUN python - <<'PYCHK'
import torch, torchvision
print('base torch', torch.__version__, 'cuda', torch.version.cuda)
print('base torchvision', torchvision.__version__)
assert torch.__version__.startswith('2.8.0'), torch.__version__
assert torchvision.__version__.startswith('0.23.0'), torchvision.__version__
PYCHK
RUN grep -Ev '^(torch|torchvision|torchaudio)==|^triton' requirements.txt > /tmp/unisharp-requirements-no-torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/unisharp-requirements-no-torch.txt \
    && python -m pip install --no-cache-dir --no-deps wandb xformers==0.0.32.post2 \
    && python - <<'PYCHK'
import torch, torchvision, gsplat, xformers
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('gsplat', gsplat.__version__ if hasattr(gsplat, '__version__') else 'ok')
print('xformers', xformers.__version__)
PYCHK

RUN mkdir -p /workspace/models \
    && python - <<'PYHF'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='Insta360-Research/Unisharp',
    filename='pretained_model.pt',
    local_dir='/workspace/models',
    local_dir_use_symlinks=False,
)
print(p)
PYHF

RUN git clone --depth 1 https://github.com/nianticlabs/spz.git /workspace/spz \
    && cmake -S /workspace/spz -B /workspace/spz/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /workspace/spz/build -j$(nproc) \
    && /workspace/spz/build/spz_info --help >/dev/null 2>&1 || true

COPY handler.py /workspace/handler.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh \
    && test -f /workspace/UniSHARP/scripts/infer_unisharp.py \
    && test -f /workspace/models/pretained_model.pt \
    && test -x /workspace/spz/build/ply_to_spz \
    && date +%s > /workspace/.unisharp_serverless_ready_v11

WORKDIR /workspace
CMD ["/workspace/start.sh"]
