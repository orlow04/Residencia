#!/usr/bin/env bash
# Sementes extras dos bandits (UCB e ε-greedy), com os mesmos argumentos da execução de 20/10 (README, seção 4).
# A execução original é a semente 0 com mensagem sem sufixo ([UCB]_*.log, [epsilon-greedy]_*.log); as novas usam
# UCB_s{1..4} e epsilon-greedy_s{1..4}. Cerca de 8 h 40 min por execução.
#
# Uso (na raiz do repositório, na 4090, venv ativo; sem variáveis de ambiente dos patches):
#   SEEDS="1 2" nohup bash roda_bandits.sh UCB > band_fila1.out 2>&1 &
#   SEEDS="1 2" nohup bash roda_bandits.sh EG  > band_fila2.out 2>&1 &
#   (e SEEDS="3 4" em outras duas filas, se a máquina aguentar quatro em paralelo)
# Retomável: execuções com log completo (200 épocas) são puladas.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
SEEDS="${SEEDS:-1 2 3 4}"
METODOS="$*"
[ -n "$METODOS" ] || { echo "informe os métodos: UCB e/ou EG"; exit 1; }
ARGS="--env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --cuda 0 --loss pointneg"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX E6_P_REP E6_P_BASE

completo() {   # $1 = pasta dos logs, $2 = mensagem; sucesso se o log mais recente tem 200 épocas
  python - "$1" "$2" "$HERE" <<'PY'
import glob, sys
logs, msg, here = sys.argv[1:4]
sys.path.insert(0, here)
from table2_from_log import read_log
fs = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))
if not fs:
    sys.exit(1)
d = read_log(fs[-1])
k = len(d[d.index >= 1])
print(f"  {msg}: {k} épocas em {fs[-1].rsplit('/', 1)[1]}")
sys.exit(0 if k == 200 else 1)
PY
}

for meth in $METODOS; do
  case "$meth" in
    UCB) SCRIPT=run_linUCB.py;        PREF=UCB;            LOGS=saved_models/KuaiEnv-v0/LinUCB/logs ;;
    EG)  SCRIPT=run_epsilongreedy.py; PREF=epsilon-greedy; LOGS=saved_models/KuaiEnv-v0/EpsilonGreedy/logs ;;
    *) echo "método desconhecido: $meth (UCB ou EG)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="${PREF}_s$s"
    if completo "$LOGS" "$msg" > /dev/null 2>&1; then echo "== $msg: já feito, pulando"; continue; fi
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    python "$SCRIPT" $ARGS --seed "$s" --message "$msg" > "band_$msg.out" 2>&1 || true
    if completo "$LOGS" "$msg"; then echo "== $msg: fim $(date '+%d/%m %H:%M')"
    else echo "== $msg: INCOMPLETO; veja $W/band_$msg.out"; fi
  done
done
echo "Fila concluída: bandits ($METODOS), sementes $SEEDS"
