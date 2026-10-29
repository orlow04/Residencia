#!/usr/bin/env bash
# Passo 1: prepara o DORL-codes com versões congeladas.
# Uso: bash setup.sh [pasta_destino]    (padrão: ./dorl-workspace)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-./dorl-workspace}"
DORL_COMMIT=818c9bd79732b87168b514695bdc2ee957782edf
TIANSHOU_COMMIT=0f59e38b126f7fb7696b79e53c86cd7b321550cb

mkdir -p "$DEST" && cd "$DEST"

# 1. Código, nos commits exatos
if [ ! -d DORL-codes ]; then
  git clone https://github.com/chongminggao/DORL-codes.git
fi
git -C DORL-codes checkout -q "$DORL_COMMIT"
git -C DORL-codes submodule update --init src/tianshou   # o submódulo aponta para o fork em $TIANSHOU_COMMIT
test "$(git -C DORL-codes/src/tianshou rev-parse HEAD)" = "$TIANSHOU_COMMIT" \
  || { echo "ERRO: tianshou não está no commit esperado"; exit 1; }

# 2. Ambiente Python 3.10 isolado (usa uv se existir; senão, python3.10 -m venv)
REQ="$HERE/requirements.txt"
if [ "$(uname -s)" = "Darwin" ]; then REQ="$HERE/requirements-macos.txt"; echo "macOS detectado: usando $REQ (não testado)"; fi
if command -v uv >/dev/null 2>&1; then
  uv venv -p 3.10 .venv
  source .venv/bin/activate
  uv pip install -r "$REQ"
else
  python3.10 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r "$REQ"
fi

# 3. Registro de versões (vai para o apêndice técnico)
{
  echo "data: $(date "+%Y-%m-%dT%H:%M:%S%z")"
  echo "DORL-codes: $(git -C DORL-codes rev-parse HEAD)"
  echo "tianshou fork: $(git -C DORL-codes/src/tianshou rev-parse HEAD)"
  python --version
  if command -v uv >/dev/null 2>&1; then uv pip freeze; else pip freeze; fi   # o venv do uv não tem pip
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "sem GPU NVIDIA visível"
} > versions.txt

# 4. Checagem da GPU
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    x = torch.randn(512, 512, device="cuda")
    print(f"GPU: {name} (compute capability {cap[0]}.{cap[1]}) | matmul OK: {(x @ x).shape}")
EOF
echo "Pronto. Ative com: source $DEST/.venv/bin/activate"
