#!/usr/bin/env bash
# E3: sensibilidade à regra de saída. MBPO e DORL treinados e avaliados com outro N (--num_leave_compute), 5 sementes,
# com as trajetórias gravadas (TRAJ_LOG). Única mudança em relação ao R: o N. O argumento vai para o KuaiEnv
# (src/core/configs.py:170), e o ambiente simulado do treino usa o done desse mesmo KuaiEnv (simulated_env.py:165),
# então o N muda no treino e na avaliação. Modelo de usuário: o original (pointneg).
#
# Uso (na raiz do repositório, na 4090, venv ativo, com traj_log.patch aplicado; sem E5_RAW_WR):
#   nohup bash roda_e3.sh 1 > e3_fila1.out 2>&1 &
#   nohup bash roda_e3.sh 8 > e3_fila2.out 2>&1 &          (em paralelo; ~5,5 h cada fila)
# Sementes: SEEDS="0 1 2 3 4" (padrão). Métodos: METODOS="MBPO DORL" (padrão). Retomável: execuções com .ok são puladas.
set -euo pipefail
N="${1:?informe o N da regra de saída (ex.: 1 ou 8)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
SEEDS="${SEEDS:-0 1 2 3 4}"
METODOS="${METODOS:-MBPO DORL}"
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute $N --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg"
mkdir -p "$DEST"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR
grep -q TRAJ_LOG src/core/evaluation/evaluator.py || { echo "traj_log.patch não aplicado"; exit 1; }

for meth in $METODOS; do
  case "$meth" in
    MBPO) LV=0;    LE=0 ;;
    DORL) LV=0.05; LE=5 ;;
    *) echo "método desconhecido: $meth (MBPO ou DORL)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="E3N${N}_${meth}_s$s"
    csv="$DEST/E3N${N}_${meth}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE \
      --seed "$s" --message "$msg" > "$DEST/$msg.out" 2>&1
    if python - "$LOGS" "$msg" "$HERE" "$N" <<'EOF'
import glob, re, sys
logs, msg, here, n = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log
f = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))[-1]
# confirma que o N foi mesmo aplicado (os argumentos ficam registrados no início do log)
cab = open(f).read(20000)
m = re.search(r"num_leave_compute['\"]?\s*[:=]\s*(\d+)", cab)
if not m or m.group(1) != n:
    print(f"  {msg}: num_leave_compute no log = {m.group(1) if m else 'não encontrado'}, esperado {n}")
    sys.exit(1)
d = read_log(f)
k = len(d[d.index >= 1])
print(f"  {msg}: {k} épocas, N = {n}, em {f.rsplit('/', 1)[1]}")
sys.exit(0 if k == 200 else 1)
EOF
    then
      sha256sum "$csv" > "$csv.ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'), $(wc -l < "$csv") linhas"
    else
      echo "== $msg: INCOMPLETO ou N errado; veja $DEST/$msg.out"
    fi
  done
done
echo "Fila concluída: N = $N ($METODOS)"
