"""Gera um KuaiRec SINTÉTICO, pequeno, com o mesmo esquema de arquivos do KuaiRec oficial.

Uso: teste de fumaça da pipeline do DORL (instalação, pré-processamento, modelo de
usuário e política) em minutos, antes de baixar os dados reais.
NÃO serve para nenhum resultado: os números produzidos com estes dados não significam nada.

    python make_synthetic_kuairec.py --out DORL-codes/environments/KuaiRec/data
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--n_users", type=int, default=200)      # real: 7.176
p.add_argument("--n_items", type=int, default=300)      # real: 10.728
p.add_argument("--small_users", type=int, default=50)   # real: 1.411
p.add_argument("--small_items", type=int, default=100)  # real: 3.327
p.add_argument("--n_cats", type=int, default=31)        # real: 31
p.add_argument("--per_user", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)

# Durações por vídeo
video_duration = rng.integers(5_000, 60_000, size=a.n_items)


def interactions(users, items_per_user):
    rows, t0 = [], 1_593_878_400.0  # julho/2020, como no KuaiRec
    for u in users:
        t = t0 + rng.uniform(0, 1e5)
        for i in items_per_user(u):
            vd = int(video_duration[i])
            play = int(vd * rng.gamma(1.5, 0.6))
            t += rng.uniform(5, 300)
            dt = pd.Timestamp(t, unit="s")
            rows.append((u, i, play, vd, dt.strftime("%Y-%m-%d %H:%M:%S.000"),
                         int(dt.strftime("%Y%m%d")), t, play / vd))
    return pd.DataFrame(rows, columns=["user_id", "video_id", "play_duration", "video_duration",
                                       "time", "date", "timestamp", "watch_ratio"])


# big_matrix: logs esparsos de todos os usuários
big = interactions(range(a.n_users),
                   lambda u: rng.choice(a.n_items, size=a.per_user, replace=False))
# small_matrix: bloco totalmente observado (subconjunto de usuários x itens)
small = interactions(range(a.small_users), lambda u: rng.permutation(a.small_items))
big.to_csv(os.path.join(a.out, "big_matrix.csv"), index=False)
small.to_csv(os.path.join(a.out, "small_matrix.csv"), index=False)

# item_categories: lista de 1 a 4 categorias por vídeo, com distribuição desigual (cauda longa)
pop = rng.zipf(1.6, size=10_000) % a.n_cats
feats = [str(sorted(set(rng.choice(pop, size=rng.integers(1, 5)).tolist()))) for _ in range(a.n_items)]
pd.DataFrame({"video_id": range(a.n_items), "feat": feats}).to_csv(
    os.path.join(a.out, "item_categories.csv"), index=False)

# user_features: mesmas colunas que o KuaiEnv lê
cat = lambda opts: rng.choice(opts, size=a.n_users)
uf = pd.DataFrame({
    "user_id": range(a.n_users),
    "user_active_degree": cat(["full_active", "high_active", "middle_active", "UNKNOWN"]),
    "is_lowactive_period": cat([0, 1]),
    "is_live_streamer": cat([0, 1]),
    "is_video_author": cat([0, 1]),
    "follow_user_num": rng.integers(0, 500, a.n_users),
    "follow_user_num_range": cat(["0", "(0,10]", "(10,50]", "(50,100]"]),
    "fans_user_num": rng.integers(0, 500, a.n_users),
    "fans_user_num_range": cat(["0", "[1,10)", "[10,100)"]),
    "friend_user_num": rng.integers(0, 100, a.n_users),
    "friend_user_num_range": cat(["0", "[1,5)", "[5,30)"]),
    "register_days": rng.integers(10, 2000, a.n_users),
    "register_days_range": cat(["15-30", "31-60", "61-90", "730+"]),
})
for k in range(18):
    uf[f"onehot_feat{k}"] = rng.integers(0, 5, a.n_users)
uf.to_csv(os.path.join(a.out, "user_features.csv"), index=False)

print(f"KuaiRec sintético gravado em {a.out}: big={len(big)} small={len(small)} interações")
