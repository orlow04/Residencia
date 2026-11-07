#!/usr/bin/env bash
# Repete as execuções do R/E2 com TRAJ_LOG para gravar as trajetórias de avaliação (traj_log.patch).
# O treino é determinístico: cada execução tem de reproduzir o log original da mesma semente; o script confere.
# Retomável: execuções já completas e conferidas são puladas; uma interrompida recomeça do zero
# (o CSV parcial é renomeado para .incompleto, nunca apagado).
#
# Uso (na raiz do repositório, na 4090, venv ativo, pe_log.patch e traj_log.patch aplicados):
#   nohup bash roda_trajetorias.sh MBPO DORL > traj_fila1.out 2>&1 &
#   nohup bash roda_trajetorias.sh MOPO DORL-const > traj_fila2.out 2>&1 &     (segunda fila, em paralelo)
# Métodos: MBPO MOPO DORL DORL-const. Sementes: SEEDS="0 1 2 3 4" (padrão).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
DEST="$W/trajetorias"
SEEDS="${SEEDS:-0 1 2 3 4}"
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute 4 --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg"
mkdir -p "$DEST"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG

confere() {  # confere <msg_nova> <msg_original>: 200 épocas e métricas idênticas ao log original
  python - "$LOGS" "$1" "$2" "$HERE" <<'EOF'
import glob, sys
logs, new, orig, here = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log
def completo(msg):
    for f in sorted(glob.glob(f"{logs}/[[]{msg}[]]_*.log"), reverse=True):
        d = read_log(f)
        if len(d[d.index >= 1]) == 200:
            return f, d
    return None, None
fn, dn = completo(new)
fo, do = completo(orig)
if dn is None:
    sys.exit(f"  {new}: sem log completo")
cols = [c for c in do.columns if any(k in c for k in ("R_tra", "ctr", "len_tra", "ifeat_feat"))]
if not do[cols].equals(dn[cols]):
    sys.exit(f"  {new}: DIFERENTE de {fo.rsplit('/', 1)[1]}")
print(f"  {new}: idêntico a {fo.rsplit('/', 1)[1]}")
EOF
}

for meth in "$@"; do
  case "$meth" in
    MBPO)       LV=0;    LE=0; ORIG=MBPO;      CONST="" ;;
    MOPO)       LV=0.05; LE=0; ORIG=MOPO;      CONST="" ;;
    DORL)       LV=0.05; LE=5; ORIG=DORL;      CONST="" ;;
    DORL-const) LV=0.05; LE=5; ORIG=DORLconst; CONST=1.9609 ;;
    *) echo "método desconhecido: $meth"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="${ORIG}_traj_s$s"
    csv="$DEST/$msg.csv"
    if [ -f "$csv.ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    [ -f "$csv" ] && mv "$csv" "$csv.incompleto.$(date +%s)"
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    if [ -n "$CONST" ]; then export PE_CONST="$CONST"; else unset PE_CONST; fi
    TRAJ_LOG="$csv" python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE \
      --seed "$s" --message "$msg" > "$DEST/$msg.out" 2>&1
    unset PE_CONST
    if confere "$msg" "${ORIG}_s$s"; then
      sha256sum "$csv" > "$csv.ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'), $(wc -l < "$csv") linhas"
    else
      echo "== $msg: FALHOU a conferência; veja $DEST/$msg.out"
    fi
  done
done
echo "Fila concluída: $*"
