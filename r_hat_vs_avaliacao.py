"""O r_hat do modelo de usuário prevê a recompensa usada na avaliação?

No DORL (KuaiRec, commit 818c9bd):
  - treino: o modelo de usuário prevê watch_ratio_normed, o watch_ratio padronizado POR VÍDEO
    (preprocessing_kuairec.py tira a média e divide pelo desvio padrão de cada vídeo);
    a matriz salva (matPre) é o r_hat que entra na recompensa da política;
  - avaliação: a recompensa é o watch_ratio BRUTO do small_matrix, limitado a 5 (KuaiEnv.load_mat).

O script monta a matriz de avaliação exatamente como o KuaiEnv.load_mat e compara, para cada usuário,
a ordenação dos itens por quatro critérios: r_hat; r_hat devolvido à escala original (média + dp do vídeo
calculados só no big_matrix, sem usar dados de avaliação); só a média do vídeo (sem personalização);
e o oráculo (a própria recompensa de avaliação). Análise estática: ignora sequência e regra de saída.

Só precisa de numpy e pandas. Roda em CPU (Mac).

Uso:
  python r_hat_vs_avaliacao.py --data environments/KuaiRec/data --matpre matPre.pickle --out rhat_aval.json
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--matpre", required=True)
p.add_argument("--k", type=int, default=10, help="tamanho da lista no top-k")
p.add_argument("--out", default="rhat_aval.json")
a = p.parse_args()


def load(name, cols):
    for fname, icol in ((f"{name}_processed.csv", "item_id"), (f"{name}.csv", "video_id")):
        path = os.path.join(a.data, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path, usecols=[c if c != "item_id" else icol for c in cols])
            print(f"lido: {path} ({len(df):,} linhas)")
            return df.rename(columns={icol: "item_id"})
    raise FileNotFoundError(f"{name}(_processed).csv não encontrado em {a.data}")


# 1. Matriz de avaliação, como em KuaiEnv.load_mat
small = load("small_matrix", ["user_id", "item_id", "watch_ratio"])
small.loc[small["watch_ratio"] > 5, "watch_ratio"] = 5
users, ui = np.unique(small["user_id"].to_numpy(), return_inverse=True)   # = LabelEncoder
items, ii = np.unique(small["item_id"].to_numpy(), return_inverse=True)
ev = np.zeros((len(users), len(items)))
np.add.at(ev, (ui, ii), small["watch_ratio"].to_numpy())                 # csr_matrix soma duplicatas
n_dup = len(small) - len(np.unique(ui * len(items) + ii))
n_missing = ev.size - len(np.unique(ui * len(items) + ii))               # células ausentes ficam 0
fill = small["watch_ratio"].mean()
ev[np.isnan(ev)] = fill
ev[np.isinf(ev)] = fill

rhat = np.asarray(pickle.load(open(a.matpre, "rb")), dtype=float)
if rhat.shape != ev.shape:
    raise SystemExit(f"matPre tem forma {rhat.shape}; a matriz de avaliação tem {ev.shape}")

# 2. Estatísticas por vídeo só do big_matrix (dados de treino)
big = load("big_matrix", ["item_id", "watch_ratio"])
st = big.groupby("item_id")["watch_ratio"].agg(["mean", "std"]).reindex(items)
mu = st["mean"].fillna(big["watch_ratio"].mean()).to_numpy()
sd = st["std"].fillna(big["watch_ratio"].std()).to_numpy()

scores = {
    "r_hat (o que a política vê)": rhat,
    "r_hat na escala original (média + dp do vídeo × r_hat)": mu[None, :] + sd[None, :] * rhat,
    "só a média do vídeo (sem personalização)": np.broadcast_to(mu, ev.shape),
    "oráculo (recompensa de avaliação)": ev,
}


def row_ranks(m):
    return pd.DataFrame(m).rank(axis=1).to_numpy()


def rowwise_corr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(1) / den


ev_rank = row_ranks(ev)
k = a.k
res_rank = {}
for name, s in scores.items():
    top = np.argsort(-s, axis=1)[:, :k]
    topk = np.take_along_axis(ev, top, axis=1).mean(1)
    rho = rowwise_corr(row_ranks(s), ev_rank)
    res_rank[name] = {f"recompensa_media_top{k}": round(float(topk.mean()), 4),
                      "spearman_por_usuario_mediana": round(float(np.nanmedian(rho)), 4)}
res_rank["aleatório (média de cada usuário)"] = {f"recompensa_media_top{k}": round(float(ev.mean(1).mean()), 4),
                                                 "spearman_por_usuario_mediana": 0.0}

# 3. Quanto da recompensa de avaliação é efeito do vídeo e quanto é efeito do usuário
tot = ev.var()
res = {
    "config": vars(a),
    "matriz_avaliacao": {"forma": list(ev.shape), "celulas_ausentes (valem 0)": int(n_missing),
                         "pares_duplicados (somados)": int(n_dup)},
    "escala": {"r_hat_dp": round(float(rhat.std()), 4), "avaliacao_dp": round(float(ev.std()), 4)},
    "correlacao_global_r_hat_vs_avaliacao": {
        "pearson": round(float(np.corrcoef(rhat.ravel(), ev.ravel())[0, 1]), 4),
        "spearman": round(float(pd.Series(rhat.ravel()).corr(pd.Series(ev.ravel()), method="spearman")), 4)},
    "variancia_da_avaliacao_explicada": {
        "pela_media_do_video": round(float(np.var(np.broadcast_to(ev.mean(0), ev.shape)) / tot), 4),
        "pela_media_do_usuario": round(float(np.var(np.broadcast_to(ev.mean(1, keepdims=True), ev.shape)) / tot), 4)},
    "media_do_video_big_vs_small (pearson)": round(float(np.corrcoef(mu, ev.mean(0))[0, 1]), 4),
    "ordenacao_por_usuario": res_rank,
}
print(json.dumps(res, indent=2, ensure_ascii=False))
with open(a.out, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\nsalvo em {a.out}")
