#!/usr/bin/env bash
# Junta em results/ tudo o que a 4090 produziu e que deve ir para o repositório.
# Uso (na raiz do repositório, na 4090, com o venv ativo):  bash exporta_resultados.sh
# Não copia nada grande: o registro bruto do P_E e as matrizes do modelo de usuário entram só pelo SHA-256.
set -euo pipefail
W=dorl-workspace/DORL-codes
L=$W/saved_models/KuaiEnv-v0
mkdir -p results/logs results/e1 results/pe results/ambiente

# 1. Logs de todas as execuções (políticas, bandits e modelo de usuário), inclusive as incompletas:
#    o e1_protocolos.py descarta as incompletas e as repetidas e lista quais usou.
find "$L" -path "*/logs/*" -name "*.log" -exec cp {} results/logs/ \;
echo "logs: $(ls results/logs | wc -l) arquivos em results/logs"

# 2. Saídas do E1 e a política aleatória usada nele
for f in e1_saida.txt e1_por_semente.csv e1_resumo.csv e1_posicoes.csv aleatoria.json; do
  [ -f "$W/$f" ] && cp "$W/$f" results/e1/
done

# 3. Registro do P_E: resumo (refeito a partir do arquivo bruto) e hash do arquivo bruto
PE=$(ls $W/pe_log_DORL_s2.csv $W/pe_log_DORL_s2.csv.gz 2>/dev/null | head -1 || true)
if [ -n "$PE" ]; then
  python analisa_pe_log.py "$PE" > results/pe/pe_log_resumo.txt
  sha256sum "$PE" > results/pe/pe_log_DORL_s2.sha256
  echo "P_E: resumo e hash gravados (o arquivo bruto fica na 4090: $PE)"
fi

# 4. Ambiente: commit do DORL, prova de que a única alteração é o pe_log.patch, versões e hash das matrizes
git -C "$W" rev-parse HEAD > results/ambiente/dorl_commit.txt
git -C "$W" diff > results/ambiente/dorl_codes_diff.patch
pip freeze > results/ambiente/versions_venv.txt
find "$W/saved_models" -name "*_mat*.pickle" -exec sha256sum {} \; > results/ambiente/mats.sha256

echo "Pronto. Confira com: git status --short results/"
