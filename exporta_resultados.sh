#!/usr/bin/env bash
# Junta em results/ o que a 4090 produziu e que entra no repositório: logs, marcadores das trajetórias, resumo do P_E
# e registro do ambiente. Nada grande entra: as trajetórias, o registro bruto do P_E, as matrizes do modelo de usuário
# e os checkpoints ficam na 4090 e entram só pelo SHA-256.
# Uso (na raiz do repositório, na 4090, com o venv ativo):  bash exporta_resultados.sh
# Depois, em qualquer máquina:  python verifica_resultados.py   (recalcula as tabelas a partir dos logs)
set -euo pipefail
W=dorl-workspace/DORL-codes
L=$W/saved_models/KuaiEnv-v0
mkdir -p results/logs results/pe results/ambiente results/trajetorias

# 1. Logs de todas as execuções (políticas, bandits e modelos de usuário), inclusive as incompletas e as repetidas:
#    o e1_protocolos.py descarta essas e lista quais usou.
find "$L" -path "*/logs/*" -name "*.log" -exec cp {} results/logs/ \;
echo "logs: $(find results/logs -name '*.log' | wc -l) arquivos em results/logs"

# 2. Trajetórias de avaliação: só o marcador .ok (SHA-256 do CSV conferido); os CSVs ficam na 4090.
if [ -d "$W/trajetorias" ]; then
  find "$W/trajetorias" -name "*.ok" -exec cp {} results/trajetorias/ \;
  echo "trajetórias: $(find results/trajetorias -name '*.ok' | wc -l) marcadores em results/trajetorias"
fi

# 3. Registro do P_E: resumo (refeito a partir do arquivo bruto) e hash do arquivo bruto
PE=$(ls $W/pe_log_DORL_s2.csv $W/pe_log_DORL_s2.csv.gz 2>/dev/null | head -1 || true)
if [ -n "$PE" ]; then
  python analisa_pe_log.py "$PE" > results/pe/pe_log_resumo.txt
  sha256sum "$PE" > results/pe/pe_log_DORL_s2.sha256
  echo "P_E: resumo e hash gravados (o arquivo bruto fica na 4090: $PE)"
fi

# 4. Ambiente: commit do DORL-codes, diff local (os patches deste repositório; cada um é inerte sem a sua variável de
#    ambiente), versões do venv e hash das matrizes do modelo de usuário
git -C "$W" rev-parse HEAD > results/ambiente/dorl_commit.txt
git -C "$W" diff > results/ambiente/dorl_codes_diff.patch
if command -v uv >/dev/null 2>&1; then uv pip freeze; else python -m pip freeze; fi > results/ambiente/versions_venv.txt
find "$W/saved_models" -name "*_mat*.pickle" -exec sha256sum {} \; > results/ambiente/mats.sha256

echo "Pronto. Confira com: git status --short results/   e, em qualquer máquina: python verifica_resultados.py"
