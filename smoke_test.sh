#!/usr/bin/env bash
# Teste de fumaça: roda a pipeline inteira do DORL (pré-processamento -> modelo de usuário ->
# DORL, UCB, epsilon-greedy -> resumo no formato da Tabela 2) com um KuaiRec SINTÉTICO pequeno.
# Leva poucos minutos e não precisa dos dados reais. Os números NÃO significam nada.
#
# Uso: bash smoke_test.sh dorl-workspace     (com o .venv já criado pelo setup.sh)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "${1:-./dorl-workspace}" && pwd)"
REPO="$WS/DORL-codes"
DATA="$REPO/environments/KuaiRec/data"
source "$WS/.venv/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

# Segurança: nunca sobrescrever dados reais
if [ -f "$DATA/big_matrix.csv" ] && [ ! -f "$DATA/.SYNTHETIC" ]; then
  echo "ERRO: $DATA já contém dados reais. O teste de fumaça só roda com a pasta vazia."; exit 1
fi

cleanup() {
  # Os ambientes gravam caches (matrizes de distância, durações, ids) dentro da pasta de dados.
  # Se ficassem, contaminariam a execução com os dados reais.
  rm -rf "$DATA" "$REPO/saved_models/KuaiEnv-v0"
  echo "Limpeza feita: dados sintéticos e modelos do teste removidos."
}
trap cleanup EXIT

cd "$REPO"
python "$HERE/make_synthetic_kuairec.py" --out "$DATA"
touch "$DATA/.SYNTHETIC"
python preprocessing_kuairec.py

echo "== modelo de usuário (DeepFM, 1 época, ensemble de 2)"
python run_worldModel_ensemble.py --env KuaiEnv-v0 --cuda 0 --epoch 1 --n_models 2 --loss pointneg --message smoke > /dev/null 2>&1

echo "== DORL (1 época curta)"
python run_Policy_Main.py --env KuaiEnv-v0 --seed 0 --cuda 0 --num_leave_compute 1 --leave_threshold 0 \
  --which_tracker avg --reward_handle cat --lambda_variance 0.05 --lambda_entropy 5 --window_size 3 \
  --read_message smoke --message DORL-smoke --epoch 1 --step-per-epoch 1000 > /dev/null 2>&1

for b in run_epsilongreedy.py run_linUCB.py; do
  echo "== $b (1 época)"
  python "$b" --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 1 --seed 0 --cuda 0 \
    --loss pointneg --message "smoke-$b" > /dev/null 2>&1
done

echo; echo "== Resumo do log do DORL (protocolos FB, NX_0 e NX_10)"
python "$HERE/table2_from_log.py" saved_models/KuaiEnv-v0/A2C_with_emb/logs/*DORL-smoke*.log --all-protocols
echo "TESTE DE FUMAÇA CONCLUÍDO: a pipeline roda de ponta a ponta."
