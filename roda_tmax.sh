#!/usr/bin/env bash
# Controle da varredura do survival bonus (docs/semana10.md, etapa 4): b = 0 não reproduz o MOPO em cobertura, e a única
# diferença no código entre os dois é o train_max, a escala da recompensa no estado de treino (run_Policy_Main.py:250):
# com lambda_entropy = 5 ele cresce 5·(entropy_max − entropy_min) = 8,55. O tmax.patch (TMAX_LAMBDA) troca só essa escala.
#   MOPO: MOPO (λ₁ = 0,05, λ₂ = 0) com o train_max do DORL (TMAX_LAMBDA=5). Se a hipótese vale, reproduz o SB0.
#   SB0 (opcional, sentido inverso): DORL-const com b = 0 (PE_CONST=0.2910) e o train_max do MOPO (TMAX_LAMBDA=0).
#        Se a hipótese vale, reproduz o MOPO.
# Regras: N4 (determinística original) e E6 (saída calibrada, N = 1), como no roda_bonus.sh. Trajetórias gravadas.
# Mensagens: TM_{MOPO|SB0}_{N4|E6}_s{seed}.
#
# Uso (na raiz do repositório, na 4090, venv ativo, com tmax.patch, traj_log.patch e e6_saida_prob.patch aplicados;
# o SB0 exige também o pe_log.patch):
#   nohup bash roda_tmax.sh E6 > tm_fila1.out 2>&1 &
#   nohup bash roda_tmax.sh N4 > tm_fila2.out 2>&1 &
# Variáveis: METODOS="MOPO" (padrão; "MOPO SB0" para os dois sentidos), SEEDS="0 1 2 3 4".
# Retomável: execuções com .ok são puladas.
set -euo pipefail
REGRA="${1:?informe a regra: N4 ou E6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
SEEDS="${SEEDS:-0 1 2 3 4}"
METODOS="${METODOS:-MOPO}"
case "$REGRA" in
  N4) N=4; EXTRA="" ;;
  E6) N=1; EXTRA="E6_P_REP=0.063 E6_P_BASE=0.026" ;;
  *) echo "regra desconhecida: $REGRA (N4 ou E6)"; exit 1 ;;
esac
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute $N --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg --lambda_variance 0.05"
mkdir -p "$DEST"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX E6_P_REP E6_P_BASE TMAX_LAMBDA
grep -q _TMAX_LAMBDA run_Policy_Main.py || { echo "tmax.patch não aplicado"; exit 1; }
grep -q TRAJ_LOG src/core/evaluation/evaluator.py || { echo "traj_log.patch não aplicado"; exit 1; }
[ "$REGRA" = N4 ] || grep -q _E6_P environments/KuaiRec/env/KuaiEnv.py || { echo "e6_saida_prob.patch não aplicado"; exit 1; }

for meth in $METODOS; do
  case "$meth" in
    MOPO) LE=0; TM=5; CONST="" ;;
    SB0)  LE=5; TM=0; CONST="PE_CONST=0.2910"
          grep -q _PE_CONST src/core/worldModel/simulated_env.py || { echo "pe_log.patch não aplicado"; exit 1; } ;;
    *) echo "método desconhecido: $meth (MOPO ou SB0)"; exit 1 ;;
  esac
  tag="TM_${meth}_${REGRA}"
  for s in $SEEDS; do
    msg="${tag}_s$s"
    csv="$DEST/${tag}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M'), lambda_entropy = $LE, TMAX_LAMBDA = $TM"
    env $EXTRA $CONST TMAX_LAMBDA="$TM" TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_entropy $LE \
      --seed "$s" --message "$msg" > "$DEST/$msg.out" 2>&1 || true
    if python - "$LOGS" "$msg" "$HERE" "$N" "$DEST/$msg.out" <<'PY'
import glob, re, sys
logs, msg, here, n, saida = sys.argv[1:6]
sys.path.insert(0, here)
from table2_from_log import read_log
fs = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))
if not fs:
    sys.exit(1)
f = fs[-1]
txt = open(f).read()
m = re.search(r"num_leave_compute['\"]?\s*[:=]\s*(\d+)", txt[:20000])
if not m or m.group(1) != n:
    print(f"  {msg}: num_leave_compute no log = {m.group(1) if m else 'não encontrado'}, esperado {n}")
    sys.exit(1)
tm = re.search(r"TMAX_LAMBDA = .*", txt) or re.search(r"TMAX_LAMBDA = .*", open(saida, errors="replace").read())
if not tm:
    print(f"  {msg}: linha do TMAX_LAMBDA não encontrada no log nem em {saida}: o train_max não foi trocado")
    sys.exit(1)
d = read_log(f)
k = len(d[d.index >= 1])
print(f"  {msg}: {k} épocas, N = {n}, em {f.rsplit('/', 1)[1]}; {tm.group(0).strip()}")
sys.exit(0 if k == 200 else 1)
PY
    then
      sha256sum "$csv" > "$csv.ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'), $(wc -l < "$csv") linhas"
    else
      echo "== $msg: INCOMPLETO, N errado ou train_max não trocado; veja $DEST/$msg.out"
    fi
  done
done
echo "Fila concluída: controle do train_max ($METODOS), regra $REGRA"
