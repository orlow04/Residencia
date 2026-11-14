#!/usr/bin/env bash
# OPE: retreina MBPO, MOPO, DORL e DORL-const com checkpoint (--is_save), mesmos argumentos do R/E2, para avaliar as
# políticas off-policy (ope.py). O save_model_fn (util/utils.py:147) grava um .pt por época; o script mantém só o da
# última época (e200) e confere que o log é idêntico ao da execução original (o treino é determinístico: DORL_pelog_s2
# = DORL_s2), comparando as médias das 200 épocas em todas as colunas de NX_0, NX_10 e FB.
# Mensagens: CK_{MBPO,MOPO,DORL,DORLconst}_s{seed}. Checkpoints em saved_models/KuaiEnv-v0/A2C_with_emb/A2C_with_emb_<msg>-e<época>.pt (policy_utils.py:35).
#
# Uso (na raiz do repositório, na 4090, venv ativo; DORLconst exige pe_log.patch; sem TRAJ_LOG/E5/A5/E6):
#   METODOS="MBPO MOPO" nohup bash roda_ckpt.sh > ck_fila1.out 2>&1 &
#   METODOS="DORL DORLconst" nohup bash roda_ckpt.sh > ck_fila2.out 2>&1 &
# Variáveis: METODOS="MBPO MOPO DORL DORLconst" (padrão), SEEDS="0 1 2 3 4". Retomável: execuções conferidas são puladas.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
MODELS="$W/saved_models/KuaiEnv-v0/A2C_with_emb"
SEEDS="${SEEDS:-0 1 2 3 4}"
METODOS="${METODOS:-MBPO MOPO DORL DORLconst}"
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute 4 --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg --is_save"
cd "$W"
unset PE_LOG PE_CONST TRAJ_LOG E5_RAW_WR A5_FIX E6_P_REP E6_P_BASE

for meth in $METODOS; do
  case "$meth" in
    MBPO)      LV=0;    LE=0; CONST="" ;;
    MOPO)      LV=0.05; LE=0; CONST="" ;;
    DORL)      LV=0.05; LE=5; CONST="" ;;
    DORLconst) LV=0.05; LE=5; CONST=1.9609
               grep -q _PE_CONST src/core/worldModel/simulated_env.py || { echo "pe_log.patch não aplicado"; exit 1; } ;;
    *) echo "método desconhecido: $meth (MBPO, MOPO, DORL ou DORLconst)"; exit 1 ;;
  esac
  for s in $SEEDS; do
    msg="CK_${meth}_s$s"
    ok="$MODELS/$msg.ok"
    if [ -f "$ok" ]; then echo "== $msg: já feito, pulando"; continue; fi
    echo "== $msg: início $(date '+%d/%m %H:%M')"
    if [ -n "$CONST" ]; then export PE_CONST="$CONST"; else unset PE_CONST; fi
    python run_Policy_Main.py $ARGS --lambda_variance $LV --lambda_entropy $LE --seed "$s" --message "$msg" \
      > "$W/ck_$msg.out" 2>&1 || true
    unset PE_CONST
    if python - "$LOGS" "$msg" "${meth}_s$s" "$HERE" <<'PY'
import glob, sys
logs, msg, orig, here = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log
def ultimo_completo(m):
    for f in sorted(glob.glob(f"{logs}/[[]{m}[]]_*.log"), reverse=True):
        d = read_log(f)
        if len(d[d.index >= 1]) == 200:
            return f, d[d.index >= 1]
    return None, None
fn, dn = ultimo_completo(msg)
fo, do = ultimo_completo(orig)
if dn is None or do is None:
    print(f"  {msg}: log incompleto ou original {orig} não encontrado"); sys.exit(1)
cols = sorted(set(dn.columns) & set(do.columns))
dif = max(abs(dn[c].mean() - do[c].mean()) for c in cols)
print(f"  {msg}: 200 épocas; maior diferença contra {fo.rsplit('/', 1)[1]} em {len(cols)} colunas = {dif:.2e}")
sys.exit(0 if dif < 1e-9 else 2)
PY
    then
      ult=$(ls "$MODELS"/*"_$msg-e"*.pt 2>/dev/null | sed 's/.*-e\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
      [ -n "$ult" ] || { echo "== $msg: nenhum checkpoint em $MODELS"; continue; }
      ls "$MODELS"/*"_$msg-e"*.pt | grep -v -F "$ult" | xargs -r rm -f
      sha256sum "$ult" > "$ok"
      echo "== $msg: fim $(date '+%d/%m %H:%M'); idêntico ao original; mantido $(basename "$ult")"
    else
      echo "== $msg: NÃO conferido (incompleto ou diferente do original); checkpoints mantidos; veja $W/ck_$msg.out"
    fi
  done
done
echo "Fila concluída: checkpoints ($METODOS), sementes $SEEDS"
