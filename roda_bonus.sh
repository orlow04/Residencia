#!/usr/bin/env bash
# Varredura do survival bonus: DORL-const (pe_log.patch, PE_CONST) com outros valores da constante c. O MIN_R não depende
# do PE_CONST (simulated_env.py:81 usa entropy_min = 0,291), então cada passo recebe o bônus fixo b = λ₂·(c − 0,291),
# com λ₂ = 5 e λ₁ = 0,05, como no DORL. b = 0 dá a mesma recompensa do MOPO (só o MAX_R, usado na normalização do
# estado, difere); b = 8,35 (c = 1,9609) é o DORL-const do E2/E6, já rodado.
#
# Regras: N4 = regra determinística original (N = 4, como o R e o E2); E6 = saída probabilística calibrada (N = 1,
# P_REP = 0,063, P_BASE = 0,026, como o roda_e6.sh). Trajetórias gravadas (TRAJ_LOG) para bolha/alem/fb/revisitas.
# Mensagens: SB{b}_{N4|E6}_s{seed}, com o ponto do b trocado por "p" (SB0p5_E6_s0).
#
# Uso (na raiz do repositório, na 4090, venv ativo, com pe_log.patch, traj_log.patch e e6_saida_prob.patch aplicados):
#   BONUS="0 0.5" nohup bash roda_bonus.sh E6 > sb_fila1.out 2>&1 &
#   BONUS="1 2 4" nohup bash roda_bonus.sh E6 > sb_fila2.out 2>&1 &
#   BONUS="0 0.5" nohup bash roda_bonus.sh N4 > sb_fila3.out 2>&1 &
#   BONUS="1 2 4" nohup bash roda_bonus.sh N4 > sb_fila4.out 2>&1 &
# Variáveis: BONUS="0 0.5 1 2 4" (padrão), SEEDS="0 1 2 3 4". Retomável: execuções com .ok são puladas.
set -euo pipefail
REGRA="${1:?informe a regra: N4 ou E6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
SEEDS="${SEEDS:-0 1 2 3 4}"
BONUS="${BONUS:-0 0.5 1 2 4}"
case "$REGRA" in
  N4) N=4; EXTRA="" ;;
  E6) N=1; EXTRA="E6_P_REP=0.063 E6_P_BASE=0.026" ;;
  *) echo "regra desconhecida: $REGRA (N4 ou E6)"; exit 1 ;;
esac
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute $N --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg --lambda_variance 0.05 --lambda_entropy 5"
mkdir -p "$DEST"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX E6_P_REP E6_P_BASE
grep -q TRAJ_LOG src/core/evaluation/evaluator.py || { echo "traj_log.patch não aplicado"; exit 1; }
grep -q _PE_CONST src/core/worldModel/simulated_env.py || { echo "pe_log.patch não aplicado"; exit 1; }
[ "$REGRA" = N4 ] || grep -q _E6_P environments/KuaiRec/env/KuaiEnv.py || { echo "e6_saida_prob.patch não aplicado"; exit 1; }

for b in $BONUS; do
  c=$(python -c "print(f'{0.291 + $b / 5:.4f}')")
  tag="SB${b/./p}_${REGRA}"
  for s in $SEEDS; do
    msg="${tag}_s$s"
    csv="$DEST/${tag}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M'), b = $b, PE_CONST = $c"
    env $EXTRA PE_CONST="$c" TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --seed "$s" --message "$msg" \
      > "$DEST/$msg.out" 2>&1 || true
    if python - "$LOGS" "$msg" "$HERE" "$N" <<'PY'
import glob, re, sys
logs, msg, here, n = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log
fs = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))
if not fs:
    sys.exit(1)
f = fs[-1]
m = re.search(r"num_leave_compute['\"]?\s*[:=]\s*(\d+)", open(f).read(20000))
if not m or m.group(1) != n:
    print(f"  {msg}: num_leave_compute no log = {m.group(1) if m else 'não encontrado'}, esperado {n}")
    sys.exit(1)
d = read_log(f)
k = len(d[d.index >= 1])
print(f"  {msg}: {k} épocas, N = {n}, em {f.rsplit('/', 1)[1]}")
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
echo "Fila concluída: survival bonus ($BONUS), regra $REGRA"
