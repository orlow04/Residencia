#!/usr/bin/env bash
# E6: MBPO e DORL com a saída probabilística calibrada nas sessões reais (e6_saida_prob.patch; sessoes_reais.py), 5 sementes,
# com as trajetórias gravadas (TRAJ_LOG). Em vez de "repete categoria da janela → sai", o usuário sai com probabilidade
# P_REP se repete e P_BASE se não repete. Valores padrão: N = 1, P_REP = 0,063, P_BASE = 0,026 (results/sessoes:
# cadeias contíguas, folga de 2 s, sessão encerrada após 30 min). Como no E3, vale no treino e na avaliação.
#
# Uso (na raiz do repositório, na 4090, venv ativo, com traj_log.patch e e6_saida_prob.patch aplicados; sem E5_RAW_WR):
#   nohup bash roda_e6.sh MBPO > e6_fila1.out 2>&1 &
#   nohup bash roda_e6.sh DORL > e6_fila2.out 2>&1 &          (em paralelo; ~3 h cada fila)
# Variáveis: SEEDS="0 1 2 3 4", N=1, P_REP=0.063, P_BASE=0.026. Retomável: execuções com .ok são puladas.
set -euo pipefail
N="${N:-1}"
P_REP="${P_REP:-0.063}"
P_BASE="${P_BASE:-0.026}"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
SEEDS="${SEEDS:-0 1 2 3 4}"
METODOS="$*"
[ -n "$METODOS" ] || { echo "informe os métodos: MBPO e/ou DORL"; exit 1; }
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute $N --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg"
mkdir -p "$DEST"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX E6_P_REP E6_P_BASE
grep -q TRAJ_LOG src/core/evaluation/evaluator.py || { echo "traj_log.patch não aplicado"; exit 1; }
grep -q _E6_P environments/KuaiRec/env/KuaiEnv.py || { echo "e6_saida_prob.patch não aplicado"; exit 1; }

for meth in $METODOS; do
  case "$meth" in
    MBPO) LV=0;    LE=0 ;;
    DORL) LV=0.05; LE=5 ;;
    *) echo "método desconhecido: $meth (MBPO ou DORL)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="E6_${meth}_s$s"
    csv="$DEST/E6_${meth}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    E6_P_REP="$P_REP" E6_P_BASE="$P_BASE" TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE \
      --seed "$s" --message "$msg" > "$DEST/$msg.out" 2>&1
    if python - "$LOGS" "$msg" "$HERE" "$N" <<'PY'
import glob, re, sys
logs, msg, here, n = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log
f = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))[-1]
cab = open(f).read(20000)
m = re.search(r"num_leave_compute['\"]?\s*[:=]\s*(\d+)", cab)
if not m or m.group(1) != n:
    print(f"  {msg}: num_leave_compute no log = {m.group(1) if m else 'não encontrado'}, esperado {n}")
    sys.exit(1)
d = read_log(f)
k = len(d[d.index >= 1])
print(f"  {msg}: {k} épocas, N = {n}, saída probabilística, em {f.rsplit('/', 1)[1]}")
sys.exit(0 if k == 200 else 1)
PY
    then
      sha256sum "$csv" > "$csv.ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'), $(wc -l < "$csv") linhas"
    else
      echo "== $msg: INCOMPLETO ou N errado; veja $DEST/$msg.out"
    fi
  done
done
echo "Fila concluída: E6 ($METODOS), N = $N, P_REP = $P_REP, P_BASE = $P_BASE"
