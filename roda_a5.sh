#!/usr/bin/env bash
# A5: MBPO e DORL com a janela da regra de saída corrigida (a5_janela.patch, ativado por A5_FIX=1), 5 sementes, com as
# trajetórias gravadas (TRAJ_LOG). Única mudança em relação ao R (N = 4) e ao E3 (N = 8): a janela compara os min(t, N)
# itens anteriores, sem o recorte do índice negativo (no original, N/2 < t < N comparam menos itens). Como no E3, a
# regra vale no treino e na avaliação (o ambiente simulado usa o done do KuaiEnv). Com N ≤ 2 não há recorte: o
# original já é a versão corrigida, então N = 1 (E3) e N = 2 não precisam desta correção.
#
# Uso (na raiz do repositório, na 4090, venv ativo, com traj_log.patch e a5_janela.patch aplicados; sem E5_RAW_WR):
#   nohup bash roda_a5.sh 4 > a5_fila1.out 2>&1 &
#   nohup bash roda_a5.sh 8 > a5_fila2.out 2>&1 &          (em paralelo; ~5,5 h cada fila)
# Sementes: SEEDS="0 1 2 3 4" (padrão). Métodos: METODOS="MBPO DORL" (padrão). Retomável: execuções com .ok são puladas.
set -euo pipefail
N="${1:?informe o N da regra de saída (ex.: 4 ou 8)}"
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
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX
grep -q TRAJ_LOG src/core/evaluation/evaluator.py || { echo "traj_log.patch não aplicado"; exit 1; }
grep -q _A5_FIX environments/KuaiRec/env/KuaiEnv.py || { echo "a5_janela.patch não aplicado"; exit 1; }

for meth in $METODOS; do
  case "$meth" in
    MBPO) LV=0;    LE=0 ;;
    DORL) LV=0.05; LE=5 ;;
    *) echo "método desconhecido: $meth (MBPO ou DORL)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="A5N${N}_${meth}_s$s"
    csv="$DEST/A5N${N}_${meth}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    A5_FIX=1 TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE \
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
print(f"  {msg}: {k} épocas, N = {n} (janela corrigida), em {f.rsplit('/', 1)[1]}")
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
echo "Fila concluída: N = $N, janela corrigida ($METODOS)"
