#!/usr/bin/env bash
# Prova, sem o histórico do shell, que o E2 (DORLconst_s0..4) rodou com PE_CONST=<c>.
# O treino é determinístico (MBPO-timing com 2 épocas = 2 primeiras épocas do MBPO_s2; DORL_pelog_s2 = DORL_s2),
# então repetir 2 épocas com a mesma constante tem de reproduzir exatamente as épocas 1 e 2 do log original.
# Controle: as mesmas 2 épocas sem PE_CONST (= DORL) não podem coincidir.
#
# Uso (na raiz do repositório, na 4090, com o venv ativo e o pe_log.patch aplicado):
#   bash verifica_E2.sh [c]        (padrão: 1.9609)     ~4 min na RTX 4090
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
C="${1:-1.9609}"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
OUT="$HERE/results/pe/verifica_E2.txt"
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute 4 --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg --lambda_variance 0.05 --lambda_entropy 5 --epoch 2"

cd "$W"
unset PE_LOG PE_CONST
for s in 0 1 2 3 4; do
  echo "== semente $s: 2 épocas com PE_CONST=$C"
  PE_CONST="$C" python run_Policy_Main.py $ARGS --seed "$s" --message "E2check_s$s" > /dev/null 2>&1
done
echo "== controle: semente 0 sem PE_CONST"
python run_Policy_Main.py $ARGS --seed 0 --message "E2control_s0" > /dev/null 2>&1

python - "$LOGS" "$C" "$HERE" <<'EOF' | tee "$OUT"
import glob, sys
sys.path.insert(0, sys.argv[3])
from table2_from_log import read_log
logs, c = sys.argv[1], sys.argv[2]
cols = ["NX_0_R_tra", "NX_0_ctr", "NX_0_len_tra", "NX_0_ifeat_feat", "R_tra", "NX_10_R_tra"]

def ep12(pattern):
    f = sorted(glob.glob(f"{logs}/{pattern}"))[-1]
    return f, read_log(f).loc[[1, 2], cols]

print(f"Verificação do E2: PE_CONST={c}, épocas 1 e 2, colunas {cols}\n")
ok_all = True
for s in range(5):
    fo, orig = ep12(f"[[]DORLconst_s{s}[]]_*.log")
    fn, new = ep12(f"[[]E2check_s{s}[]]_*.log")
    ok = orig.equals(new)
    ok_all &= ok
    print(f"semente {s}: {'IDÊNTICO' if ok else 'DIFERENTE'}  ({fo.rsplit('/', 1)[1]} × {fn.rsplit('/', 1)[1]})")
_, ctrl = ep12("[[]E2control_s0[]]_*.log")
_, orig0 = ep12("[[]DORLconst_s0[]]_*.log")
ctrl_ok = not ctrl.equals(orig0)
print(f"controle (sem PE_CONST, semente 0): {'difere, como esperado' if ctrl_ok else 'IGUAL: o teste não discrimina'}")
print(f"\nResultado: {'E2 rodou com PE_CONST=' + c if ok_all and ctrl_ok else 'NÃO CONFIRMADO'}")
EOF
echo "gravado em $OUT"
