"""Diagnóstico do termo de entropia P_E do DORL (KuaiRec), sem treinar nada.

Pergunta: quanto do P_E é sinal e quanto é preenchimento? No código do DORL (commit 818c9bd):
  - a tabela de entropia é construída a partir do big_matrix (user_model_ensemble.get_save_entropy_mat):
    para cada histórico (último item, k=1; par ordenado dos dois últimos, k=2), a entropia normalizada
    da distribuição do PRÓXIMO item nos logs; se só existe uma continuação observada, o valor é 1;
  - na recompensa (simulated_env._compute_pred_reward), P_E = soma sobre k=1,2 do valor da tabela;
    histórico ausente da tabela -> 1; no passo t=0 falta o par -> +1.
Este script replica essas regras, mede a cobertura da tabela sob duas políticas de referência
(uniforme e por popularidade, sobre os 3.327 itens do small_matrix) e, opcionalmente, compara a escala
de lambda_2*P_E com a de r_hat e lambda_1*P_U usando as matrizes salvas pelo modelo de usuário.

Só precisa de numpy e pandas. Roda em CPU (Mac), alguns minutos com o KuaiRec completo.

Uso:
  python diagnostico_pe.py --data <pasta com big_matrix(_processed).csv e small_matrix(_processed).csv> \
      [--matpre '[pointneg]_matPre.pickle' --matvar '[pointneg]_matVar.pickle'] --out pe_diag.json
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--matpre", default=None, help="matriz r_hat salva pelo run_worldModel_ensemble (opcional)")
p.add_argument("--matvar", default=None, help="matriz de variância máxima (opcional)")
p.add_argument("--lambda_variance", type=float, default=0.05)
p.add_argument("--lambda_entropy", type=float, default=5.0)
p.add_argument("--n_traj", type=int, default=20000)
p.add_argument("--max_turn", type=int, default=30)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="pe_diag.json")
a = p.parse_args()
rng = np.random.default_rng(a.seed)


def load(name):
    for fname, col in ((f"{name}_processed.csv", "item_id"), (f"{name}.csv", "video_id")):
        path = os.path.join(a.data, fname)
        if os.path.isfile(path):
            cols = ["user_id", col, "timestamp"] if name == "big_matrix" else [col]
            df = pd.read_csv(path, usecols=cols).rename(columns={col: "item_id"})
            print(f"lido: {path} ({len(df):,} linhas)")
            return df
    raise FileNotFoundError(f"{name}(_processed).csv não encontrado em {a.data}")


# 1. Tabela de entropia, com a mesma ordenação e as mesmas regras do código original
big = load("big_matrix")
df = big[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
u = df["user_id"].to_numpy()
it = df["item_id"].to_numpy().astype(np.int64)
M = int(it.max()) + 1
same1 = np.r_[False, u[1:] == u[:-1]]           # existe item anterior do mesmo usuário
same2 = np.r_[False, False, u[2:] == u[:-2]]    # existem dois itens anteriores
prev1 = np.r_[-1, it[:-1]]
prev2 = np.r_[-1, -1, it[:-2]]


def build_table(keys, nxt):
    """Entropia normalizada da distribuição do próximo item por chave (get_entropy do DORL)."""
    comb, cnt = np.unique(keys * M + nxt, return_counts=True)
    k = comb // M
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    key_u = k[starts]
    n = np.add.reduceat(cnt, starts)
    d = np.diff(np.r_[starts, len(k)])
    prob = cnt / np.repeat(n, d)
    h = np.add.reduceat(-prob * np.log2(prob), starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(d <= 1, 1.0, h / np.log2(np.maximum(d, 2)))
    return key_u, ent, d


def pair_key(x, y):
    return np.minimum(x, y) * M + np.maximum(x, y)


K1, E1, D1 = build_table(prev1[same1], it[same1])
K2, E2, D2 = build_table(pair_key(prev1[same2], prev2[same2]), it[same2])
entropy_min = float(E1.min() + E2.min())   # como em run_Policy_Main.py (soma dos mínimos por k)
entropy_max = float(E1.max() + E2.max())


def table_stats(E, D):
    return {
        "n_chaves": int(len(E)),
        "frac_uma_continuacao (vale 1 por regra)": round(float(np.mean(D <= 1)), 4),
        "frac_valor_igual_1": round(float(np.mean(E >= 1 - 1e-12)), 4),
        "quantis_10_50_90": [round(float(q), 4) for q in np.quantile(E, [0.1, 0.5, 0.9])],
    }


def lookup(K, E, D, q):
    idx = np.clip(np.searchsorted(K, q), 0, len(K) - 1)
    found = K[idx] == q
    val = np.where(found, E[idx], 1.0)             # ausente -> 1 (simulated_env.py, l. 147)
    single = found & (D[idx] <= 1)
    return val, found, single


# 2. Espaço de ações do ambiente: itens do small_matrix (LabelEncoder ordena os ids)
S = np.sort(load("small_matrix")["item_id"].unique()).astype(np.int64)
pop = pd.Series(it).value_counts().reindex(S, fill_value=0).to_numpy().astype(float)
policies = {"uniforme": np.full(len(S), 1 / len(S))}
if pop.sum() > 0:
    policies["popularidade"] = pop / pop.sum()

# Correlação entre popularidade do item e o termo k=1 (com preenchimento)
e1_items, _, _ = lookup(K1, E1, D1, S)
corr_pop_e1 = float(pd.Series(np.log1p(pop)).corr(pd.Series(e1_items), method="spearman"))

matpre = matvar = None
if a.matpre and a.matvar:
    matpre = np.asarray(pickle.load(open(a.matpre, "rb")), dtype=float)
    matvar = np.asarray(pickle.load(open(a.matvar, "rb")), dtype=float)
    if matpre.shape[1] != len(S):
        raise SystemExit(f"matPre tem {matpre.shape[1]} colunas, esperado {len(S)} itens do small_matrix")
    min_r = matpre.min() - a.lambda_variance * matvar.max() + a.lambda_entropy * entropy_min

res = {
    "config": vars(a),
    "tabela_k1": table_stats(E1, D1),
    "tabela_k2": table_stats(E2, D2),
    "entropy_min": entropy_min,
    "entropy_max": entropy_max,
    "spearman_log_popularidade_vs_PE_k1": round(corr_pop_e1, 4),
    "politicas": {},
}

# 3. Trajetórias de comprimento fixo (sem regra de saída): P_E por passo, como no simulated_env
T = a.max_turn
for name, prob in policies.items():
    A_idx = rng.choice(len(S), size=(a.n_traj, T), p=prob)
    A = S[A_idx]
    pe = np.zeros((a.n_traj, T))
    miss1 = np.zeros((a.n_traj, T), bool)
    miss2 = np.zeros((a.n_traj, T), bool)
    single = np.zeros((a.n_traj, T), int)
    for t in range(T):
        v1, f1, s1 = lookup(K1, E1, D1, A[:, t])
        if t == 0:
            v2, f2, s2 = np.ones(a.n_traj), np.zeros(a.n_traj, bool), np.zeros(a.n_traj, bool)
        else:
            v2, f2, s2 = lookup(K2, E2, D2, pair_key(A[:, t], A[:, t - 1]))
        pe[:, t] = v1 + v2
        miss1[:, t], miss2[:, t] = ~f1, ~f2
        single[:, t] = s1.astype(int) + s2.astype(int)
    steady = pe[:, 1:]
    out = {
        "PE_media_t0": round(float(pe[:, 0].mean()), 4),
        "PE_media_t>=1": round(float(steady.mean()), 4),
        "PE_dp_t>=1": round(float(steady.std()), 4),
        "frac_passos_PE_no_maximo_2 (t>=1)": round(float(np.mean(steady >= 2 - 1e-12)), 4),
        "frac_k1_ausente": round(float(miss1.mean()), 4),
        "frac_k2_ausente (t>=1)": round(float(miss2[:, 1:].mean()), 4),
        "frac_termos_valendo_1_por_uma_continuacao (t>=1)": round(float(single[:, 1:].mean() / 2), 4),
        "bonus_medio_por_passo lambda2*(PE - entropy_min) (t>=1)":
            round(float(a.lambda_entropy * (steady.mean() - entropy_min)), 4),
    }
    if matpre is not None:
        users = rng.integers(0, matpre.shape[0], size=a.n_traj)[:, None].repeat(T, 1)
        r_hat = matpre[users, A_idx]
        pu = a.lambda_variance * matvar[users, A_idx]
        pen = a.lambda_entropy * pe
        r_tilde = np.clip(r_hat - pu + pen - min_r, 0, None)
        out["escala (t>=1)"] = {
            nm: {"media": round(float(x[:, 1:].mean()), 4), "dp": round(float(x[:, 1:].std()), 4)}
            for nm, x in (("r_hat", r_hat), ("lambda1*P_U", pu), ("lambda2*P_E", pen), ("r_tilde", r_tilde))
        }
        out["corr_r_tilde_com_lambda2PE (t>=1)"] = round(float(np.corrcoef(
            r_tilde[:, 1:].ravel(), pen[:, 1:].ravel())[0, 1]), 4)
    res["politicas"][name] = out

print(json.dumps(res, indent=2, ensure_ascii=False))
with open(a.out, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\nsalvo em {a.out}")
