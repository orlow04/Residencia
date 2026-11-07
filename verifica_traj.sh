#!/usr/bin/env bash
# Verifica o traj_log.patch numa execução real, antes de usá-lo nas execuções longas:
#   (1) o patch é inerte: 2 épocas do MBPO_s2 com TRAJ_LOG reproduzem exatamente as épocas 1 e 2 do log original;
#   (2) o CSV reproduz as métricas do log em cada época e protocolo (R_tra, Length, R_each, MCD);
#   (3) a recompensa gravada é a da matriz de avaliação: reward == mat[user, item].
#
# Uso (na raiz do repositório, na 4090, com o venv ativo e o traj_log.patch aplicado no DORL-codes):
#   bash verifica_traj.sh        ~2 min na RTX 4090
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/dorl-workspace/DORL-codes"
LOGS="$W/saved_models/KuaiEnv-v0/A2C_with_emb/logs"
CSV="$W/traj_verifica_MBPO_s2.csv"
OUT="$HERE/results/trajetorias/verifica_traj.txt"
mkdir -p "$(dirname "$OUT")"
rm -f "$CSV"

cd "$W"
unset PE_LOG PE_CONST
echo "== MBPO semente 2, 2 épocas, com TRAJ_LOG"
TRAJ_LOG="$CSV" python run_Policy_Main.py --env KuaiEnv-v0 --seed 2 --cuda 0 --num_leave_compute 4 \
  --leave_threshold 0 --which_tracker avg --reward_handle cat --lambda_variance 0 --lambda_entropy 0 \
  --window_size 3 --read_message pointneg --epoch 2 --message TRAJcheck_s2 > /dev/null 2>&1

python - "$LOGS" "$CSV" "$HERE" "$W/environments/KuaiRec" <<'EOF' | tee "$OUT"
import ast, glob, pickle, sys
import numpy as np, pandas as pd
logs, csv, here, kr = sys.argv[1:5]
sys.path.insert(0, here)
from table2_from_log import read_log

def last(pattern):
    return read_log(sorted(glob.glob(f"{logs}/{pattern}"))[-1])

orig, new = last("[[]MBPO_s2[]]_*.log"), last("[[]TRAJcheck_s2[]]_*.log")
cols = [c for c in new.columns if any(k in c for k in ("R_tra", "ctr", "len_tra", "ifeat_feat"))]
inerte = orig.loc[[1, 2], cols].equals(new.loc[[1, 2], cols])
print(f"(1) patch inerte (épocas 1 e 2 idênticas ao [MBPO_s2] original): {'SIM' if inerte else 'NÃO'}")

df = pd.read_csv(csv)
cat = pd.read_csv(f"{kr}/data/item_categories.csv")["feat"].map(ast.literal_eval).to_list()
dom = pickle.load(open(f"{kr}/env/feature_domination.pickle", "rb"))["feat"]
v = np.array([float(x[1]) for x in dom]); cum = (v / v.sum()).cumsum()
dominated = {int(x[0]) for x in dom[:max(int(np.sum(np.cumsum(cum > 0.8) == 0)), 1)]}
df["dom"] = [any((c + 1) in dominated for c in cat[i]) for i in df["item_raw"]]

ok2 = True
print("\n(2) métricas recalculadas do CSV × log")
for (ep, proto), d in df.groupby(["epoch", "protocolo"]):
    pre = "" if proto == "FB" else proto + "_"
    n = d["traj"].nunique()
    calc = {"R_tra": d["reward"].sum() / n, "len_tra": len(d) / n,
            "ctr": d["reward"].sum() / len(d), "ifeat_feat": d["dom"].mean()}
    diffs = {k: abs(calc[k] - new.loc[ep, pre + k]) for k in calc}
    ok = all(x < (6e-6 if k == "ctr" else 1e-6) for k, x in diffs.items())   # ctr vem com 5 casas no log
    ok2 &= ok
    print(f"  época {ep} {proto:5s} n={n:3d}: {'OK' if ok else 'DIVERGE'}  "
          + "  ".join(f"{k} {calc[k]:.4f}/{new.loc[ep, pre + k]:.4f}" for k in calc))

small = pd.read_csv(f"{kr}/data/small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
small.loc[small["watch_ratio"] > 5, "watch_ratio"] = 5
_, ui = np.unique(small["user_id"], return_inverse=True)
_, ii = np.unique(small["video_id"], return_inverse=True)
mat = np.zeros((ui.max() + 1, ii.max() + 1)); np.add.at(mat, (ui, ii), small["watch_ratio"].to_numpy())
ok3 = np.allclose(df["reward"], mat[df["user"], df["item"]])
print(f"\n(3) reward == mat[user, item] em {len(df):,} passos: {'SIM' if ok3 else 'NÃO'}")
print(f"\nResultado: {'traj_log.patch VERIFICADO' if inerte and ok2 and ok3 else 'NÃO VERIFICADO'}")
EOF
echo "gravado em $OUT"
