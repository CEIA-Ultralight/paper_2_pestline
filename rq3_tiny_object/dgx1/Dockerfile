# =============================================================================
# Dockerfile — container da RQ3 para a dgx1 (8x V100 32 GB, driver 550 = CUDA <= 12.4).
#
# O flydet-train-b200.sif (torch 2.8 + cu128) nao roda com driver 550, por isso aqui:
#   - torch 2.6.0 + torchvision 0.21.0 (wheels cu124; as libs CUDA vem nos wheels)
#   - ultralytics==8.4.142 PINADO (versao contra a qual rq3_tiny_object/patches/* foram escritos)
#   - fly_det do submodule baseline/fly-det (commit pinado no repo do paper)
# O codigo do paper (rq*/, common/) NAO entra na imagem: e montado do repo em tempo de execucao.
#
# Build (na raiz do repo, com o submodule baseline inicializado):
#   docker build -f rq3_tiny_object/dgx1/Dockerfile -t pestline-rq3:cu124 .
# =============================================================================
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/Sao_Paulo

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir uv
ENV PATH="/opt/venv/bin:${PATH}"

# 1) PyTorch cu124 fixo; as constraints impedem upgrade nos passos seguintes
RUN printf '%s\n' 'torch==2.6.0' 'torchvision==0.21.0' 'ultralytics==8.4.142' > /opt/constraints.txt \
 && uv pip install --no-cache --python /opt/venv/bin/python \
        torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) fly_det (baseline pinado) + dependencias usadas por train_nwd.py e eval_cascade_matrix.py
COPY baseline/fly-det /opt/src/fly-det
RUN uv pip install --no-cache --python /opt/venv/bin/python -c /opt/constraints.txt \
        /opt/src/fly-det "ultralytics==8.4.142" "albumentations>=2.0.3" wandb pandas pyyaml \
 && python -c "import torch, ultralytics, fly_det.trainer; assert ultralytics.__version__ == '8.4.142'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ultralytics', ultralytics.__version__)"

WORKDIR /workspace
