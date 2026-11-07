#!/usr/bin/env bash
# E5: modelo de usuário treinado com o watch_ratio bruto limitado a 5 (o sinal da avaliação) no lugar do
# watch_ratio_normed; depois MBPO e DORL com esse modelo, 5 sementes, com as trajetórias gravadas (TRAJ_LOG).
# Única mudança em relação ao R: o alvo do modelo de usuário (e5_alvo.patch, ativado só por E5_RAW_WR no
# passo "modelo"). As políticas rodam SEM E5_RAW_WR e leem o modelo novo com --read_message rawwr.
#
# Uso (na raiz do repositório, na 4090, venv ativo, com pe_log.patch, traj_log.patch e e5_alvo.patch aplicados;
# NÃO rodar enquanto houver execuções de política em andamento: o passo "modelo" regrava entropy/map_entropy.pickle,
# que toda execução lê ao iniciar):
#   bash roda_e5.sh modelo                                    ~55 min; uma vez só
#   nohup bash roda_e5.sh MBPO > e5_fila1.out 2>&1 &
#   nohup bash roda_e5.sh DORL > e5_fila2.out 2>&1 &          (em paralelo)
# Sementes: SEEDS="0 1 2 3 4" (padrão). Retomável: execuções com .ok são puladas.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
UM="$W/saved_models/KuaiEnv-v0/DeepFM"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
OUT="$HERE/results/e5"
SEEDS="${SEEDS:-0 1 2 3 4}"
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute 4 --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message rawwr"
mkdir -p "$DEST" "$OUT"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR
grep -q E5_RAW_WR environments/KuaiRec/env/KuaiEnv.py || { echo "e5_alvo.patch não aplicado"; exit 1; }

if [ "${1:-}" = modelo ]; then
  [ -f environments/KuaiRec/data/video_duration_normed.csv ] || \
    echo "AVISO: sem video_duration_normed.csv; o carregador vai criá-lo agora (o pointneg pode ter usado a versão em memória)"
  if pgrep -f run_Policy_Main.py > /dev/null; then echo "há execuções de política em andamento; espere terminarem"; exit 1; fi
  ENT="$UM/entropy/map_entropy.pickle"
  [ -f "$ENT.pointneg" ] || cp "$ENT" "$ENT.pointneg"
  if [ -f "$UM/matsPre/[rawwr]_matPre.pickle" ]; then echo "modelo rawwr já existe; nada a fazer"; exit 0; fi
  echo "== modelo de usuário rawwr: início $(date '+%d/%m %H:%M')"
  E5_RAW_WR=1 python run_worldModel_ensemble.py --env KuaiEnv-v0 --cuda 0 --epoch 5 --loss pointneg \
    --message rawwr > "$OUT/modelo_rawwr.out" 2>&1
  # A tabela de entropia só depende da ordem dos itens no log, não do alvo: tem de sair igual à do pointneg.
  if python - "$ENT" "$ENT.pointneg" <<'EOF'
import pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:3])
sys.exit(0 if a == b else 1)
EOF
  then echo "  tabela de entropia idêntica à do pointneg"
  else cp "$ENT.pointneg" "$ENT"; echo "  ERRO: tabela de entropia mudou; a original foi restaurada"; exit 1; fi
  python - "$UM" <<'EOF' | tee "$OUT/modelo_rawwr_resumo.txt"
import pickle, sys
import numpy as np
um = sys.argv[1]
for msg in ("pointneg", "rawwr"):
    p = pickle.load(open(f"{um}/matsPre/[{msg}]_matPre.pickle", "rb"))
    v = pickle.load(open(f"{um}/matsVar/[{msg}]_matVar.pickle", "rb"))
    print(f"{msg:8s} r_hat: média {p.mean():.3f}, dp {p.std():.3f}, mín {p.min():.3f}, máx {p.max():.3f}; "
          f"P_U máx {v.max():.3f}; forma {p.shape}")
EOF
  sha256sum "$UM"/matsPre/"[rawwr]_matPre.pickle" "$UM"/matsVar/"[rawwr]_matVar.pickle" > "$OUT/mats_rawwr.sha256"
  echo "== modelo de usuário rawwr: fim $(date '+%d/%m %H:%M')"
  exit 0
fi

[ -f "$UM/matsPre/[rawwr]_matPre.pickle" ] || { echo "rode antes: bash roda_e5.sh modelo"; exit 1; }
for meth in "$@"; do
  case "$meth" in
    MBPO) LV=0;    LE=0 ;;
    DORL) LV=0.05; LE=5 ;;
    *) echo "método desconhecido: $meth (MBPO ou DORL)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="E5_${meth}_s$s"
    csv="$DEST/E5_${meth}_traj_s$s.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE \
      --seed "$s" --message "$msg" > "$DEST/$msg.out" 2>&1
    if python - "$LOGS" "$msg" "$HERE" <<'EOF'
import glob, sys
logs, msg, here = sys.argv[1:4]
sys.path.insert(0, here)
from table2_from_log import read_log
f = sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"))[-1]
d = read_log(f)
n = len(d[d.index >= 1])
print(f"  {msg}: {n} épocas em {f.rsplit('/', 1)[1]}")
sys.exit(0 if n == 200 else 1)
EOF
    then
      sha256sum "$csv" > "$csv.ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'), $(wc -l < "$csv") linhas"
    else
      echo "== $msg: INCOMPLETO; veja $DEST/$msg.out"
    fi
  done
done
echo "Fila concluída: $*"
