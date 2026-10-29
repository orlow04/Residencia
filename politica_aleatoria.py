"""Política aleatória uniforme avaliada no ambiente de avaliação do DORL (KuaiRec), nos três protocolos.

Serve de referência para R_each, R_tra, Length e MCD: quanto um recomendador que NÃO usa nenhum sinal
consegue em cada métrica. Replica do DORL-codes (commit 818c9bd):
  - matriz de recompensa: KuaiEnv.load_mat (watch_ratio bruto do small_matrix, limitado a 5; ausente = 0);
  - usuário sorteado uniformemente a cada trajetória; max_turn = 30;
  - regra de saída: KuaiEnv._determine_whether_to_leave (inclusive o recorte da janela nos passos iniciais);
  - protocolos (collector_set.py): FB repete itens; NX_0 não repete; NX_10 não repete e força 10 rodadas;
  - métricas (util/utils.py e evaluator.py), por época de 100 trajetórias:
      Length = passos / trajetórias; R_tra = recompensa total / trajetórias; R_each = R_tra / Length;
      MCD = fração dos itens recomendados com alguma categoria dominante (feature_domination.pickle, top 80%);
  - agregação: média ± dp sobre as épocas, como na Tabela 2.

Só precisa de numpy e pandas. Roda em CPU (Mac), cerca de 1 minuto.

Uso:
  python politica_aleatoria.py --data environments/KuaiRec/data \
      --domination environments/KuaiRec/env/feature_domination.pickle --seeds 0 1 2 3 4 --out aleatoria.json
"""
import argparse
import ast
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--domination", required=True, help="environments/KuaiRec/env/feature_domination.pickle")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--n_traj", type=int, default=100)
p.add_argument("--num_leave_compute", type=int, default=4)
p.add_argument("--leave_threshold", type=float, default=0)
p.add_argument("--max_turn", type=int, default=30)
p.add_argument("--force_length", type=int, default=10)
p.add_argument("--top_rate", type=float, default=0.8)
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--out", default="aleatoria.json")
a = p.parse_args()

# 1. Matriz de recompensa (KuaiEnv.load_mat)
path = next(os.path.join(a.data, f) for f in ("small_matrix_processed.csv", "small_matrix.csv")
            if os.path.isfile(os.path.join(a.data, f)))
small = pd.read_csv(path)
small = small.rename(columns={"video_id": "item_id"})[["user_id", "item_id", "watch_ratio"]]
small.loc[small["watch_ratio"] > 5, "watch_ratio"] = 5
users, ui = np.unique(small["user_id"].to_numpy(), return_inverse=True)
items, ii = np.unique(small["item_id"].to_numpy(), return_inverse=True)
mat = np.zeros((len(users), len(items)))
np.add.at(mat, (ui, ii), small["watch_ratio"].to_numpy())
fill = small["watch_ratio"].mean()
mat[np.isnan(mat)] = fill
mat[np.isinf(mat)] = fill
n_users, n_items = mat.shape
print(f"lido: {path}; matriz {n_users} x {n_items}")

# 2. Categorias (KuaiEnv.load_category): lista original para a regra de saída; +1 para a MCD
cat = pd.read_csv(os.path.join(a.data, "item_categories.csv"))
if not np.array_equal(cat["video_id"].to_numpy(), np.arange(len(cat))):
    print("AVISO: video_id != posição da linha em item_categories.csv; o DORL indexa pela posição")
list_feat = cat["feat"].map(ast.literal_eval).to_list()          # o DORL indexa pela posição da linha
list_feat_small = [list(list_feat[i]) for i in items]

dom = pickle.load(open(a.domination, "rb"))["feat"]
v = np.array([float(x[1]) for x in dom])
cum = (v / v.sum()).cumsum()
ind = max(int(np.sum(np.cumsum(cum > a.top_rate) == 0)), 1)    # mesmo laço de get_feat_dominate_dict
dominated = {int(x[0]) for x in dom[:ind]}
is_dom = np.array([any((c + 1) in dominated for c in feats) for feats in list_feat_small])
print(f"categorias dominantes (codificação +1 do DORL): {sorted(dominated)} ({len(dominated)})")


def leave(seq, t, action):
    """Cópia de KuaiEnv._determine_whether_to_leave (seq = ações ANTES do passo t)."""
    if t == 0:
        return False
    window = seq[t - a.num_leave_compute:t]
    hist = Counter(c for x in window for c in list_feat_small[x])
    return any(hist[c] > a.leave_threshold for c in list_feat_small[action])


def trajectory(rng, protocol):
    user = int(rng.integers(0, n_users))
    seq, rew = [], 0.0
    force = a.force_length if protocol == "NX_10" else 0
    used = set()
    while True:
        t = len(seq)
        if protocol == "FB":
            action = int(rng.integers(0, n_items))
        else:                                    # NX: não repete itens da trajetória
            action = int(rng.integers(0, n_items))
            while action in used:
                action = int(rng.integers(0, n_items))
        done = False if force else (leave(seq, t, action) or t >= a.max_turn - 1)
        seq.append(action)
        used.add(action)
        rew += mat[user, action]
        if force:
            if len(seq) >= force:
                break
        elif done:
            break
    return seq, rew


res = {"config": vars(a), "categorias_dominantes": sorted(dominated), "por_semente": {}}
summary_rows = []
for seed in a.seeds:
    rng = np.random.default_rng(seed)
    out = {}
    for protocol in ("FB", "NX_0", "NX_10"):
        ep = []
        for _ in range(a.epochs):
            trajs = [trajectory(rng, protocol) for _ in range(a.n_traj)]
            steps = sum(len(s) for s, _ in trajs)
            r_tra = sum(r for _, r in trajs) / a.n_traj
            length = steps / a.n_traj
            mcd = float(np.mean(is_dom[np.concatenate([s for s, _ in trajs])]))
            ep.append((r_tra, r_tra / length, length, mcd))
        ep = np.array(ep)
        out[protocol] = {m: [round(float(ep[:, j].mean()), 4), round(float(ep[:, j].std(ddof=1)), 4)]
                         for j, m in enumerate(["R_tra", "R_each", "Length", "MCD"])}
        summary_rows += [{"semente": seed, "protocolo": protocol, "metrica": m, "valor": out[protocol][m][0]}
                         for m in out[protocol]]
    res["por_semente"][seed] = out

df = pd.DataFrame(summary_rows)
agg = df.groupby(["protocolo", "metrica"])["valor"].agg(["mean", "std", "count"]).round(4)
res["entre_sementes"] = {f"{pr}:{m}": list(map(float, agg.loc[(pr, m)])) for pr, m in agg.index}
print("\nPolítica aleatória uniforme — média entre sementes (dp entre sementes; n sementes)")
print(agg.rename(columns={"mean": "média", "std": "dp", "count": "n"}).to_string())
with open(a.out, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\nsalvo em {a.out}")
