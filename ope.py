"""Avaliação off-policy (OPE) por passo: IPS, SNIPS, IPS com corte e doubly robust (DR), e a validação semi-sintética.

Cenário de bandit contextual por passo: dados registrados (x, a, r) com a ~ π_β(·|x); o valor da política-alvo π_e é
V(π_e) = E_x Σ_a π_e(a|x) r(x, a). Com w = π_e(a|x) / π_β(a|x):
  IPS      V̂ = média(w·r)                                  (sem viés se π_β > 0 onde π_e > 0)
  SNIPS    V̂ = Σ w·r / Σ w                                 (autonormalizado; viés pequeno, variância menor)
  IPS-M    V̂ = média(min(w, M)·r)                          (corte dos pesos; viés para baixo)
  DM       V̂ = média(Σ_a π_e(a|x) q̂(x, a))                 (modelo direto, sem pesos)
  DR       V̂ = média(Σ_a π_e(a|x) q̂(x, a) + w·(r − q̂(x, a)))   (sem viés se π_β ou q̂ estiver correto)
Diagnóstico: tamanho efetivo da amostra ESS = (Σ w)² / Σ w² e o maior peso.

Validação semi-sintética (--validacao): o small_matrix é totalmente observado, então o valor verdadeiro de qualquer
política é conhecido. O contexto é o usuário (x = u, um passo). Gera-se um log com uma política de registro conhecida,
proporcional à popularidade dos itens no big_matrix (usuários fora da avaliação) misturada com ε de uniforme, e mede-se
o viés e o RMSE de cada estimador contra o valor verdadeiro, com π_β conhecida e com π_β estimada pelas contagens do
próprio log (o caso real, em que π_β é desconhecida). As políticas-alvo variam a concentração, como as do RL:
uniforme; k itens fixos ao acaso, iguais para todos os usuários (k = 300, 100, 30, 10); os k mais populares; e o
oráculo top-10 por usuário (personalizado).

Uso (de dorl-workspace/DORL-codes):
  python ../../ope.py --validacao --data environments/KuaiRec/data --out ../../results/ope
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


# ---------- estimadores ----------

def pesos(pe_a, pb_a):
    """w = π_e(a|x) / π_β(a|x) para as ações registradas."""
    return np.asarray(pe_a, float) / np.asarray(pb_a, float)


def ips(w, r):
    return float(np.mean(w * r))


def snips(w, r):
    return float(np.sum(w * r) / np.sum(w)) if np.sum(w) > 0 else np.nan


def ips_corte(w, r, m):
    return float(np.mean(np.minimum(w, m) * r))


def dm(v_q):
    """v_q[i] = Σ_a π_e(a|x_i) q̂(x_i, a)."""
    return float(np.mean(v_q))


def dr(w, r, q_a, v_q):
    """q_a[i] = q̂(x_i, a_i); v_q[i] = Σ_a π_e(a|x_i) q̂(x_i, a)."""
    return float(np.mean(v_q + w * (r - q_a)))


def ess(w):
    return float(np.sum(w) ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0


# ---------- dados ----------

def matriz_avaliacao(data):
    """Recompensa de avaliação como KuaiEnv.load_mat: watch_ratio do small_matrix limitado a 5; ausente = 0."""
    path = next(os.path.join(data, f) for f in ("small_matrix_processed.csv", "small_matrix.csv")
                if os.path.isfile(os.path.join(data, f)))
    s = pd.read_csv(path).rename(columns={"video_id": "item_id"})[["user_id", "item_id", "watch_ratio"]]
    s.loc[s["watch_ratio"] > 5, "watch_ratio"] = 5
    users, ui = np.unique(s["user_id"].to_numpy(), return_inverse=True)
    items, ii = np.unique(s["item_id"].to_numpy(), return_inverse=True)
    mat = np.zeros((len(users), len(items)))
    np.add.at(mat, (ui, ii), s["watch_ratio"].to_numpy())
    fill = s["watch_ratio"].mean()
    mat[~np.isfinite(mat)] = fill
    return mat, users, items


def popularidade_externa(data, users, items):
    """Visualizações de cada item do small_matrix no big_matrix, só usuários fora da avaliação (sem duplicatas)."""
    b = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    b = b.drop_duplicates()
    b = b[~b["user_id"].isin(set(users)) & b["video_id"].isin(set(items))]
    return b["video_id"].value_counts().reindex(items, fill_value=0).to_numpy().astype(float)


# ---------- validação semi-sintética ----------

def alvos(mat, pop, rng):
    """Políticas-alvo como matrizes π_e (usuários × itens), linhas somando 1."""
    n_u, n_i = mat.shape
    out = {"uniforme": np.full((n_u, n_i), 1 / n_i)}
    for k in (300, 100, 30, 10):
        p = np.zeros(n_i)
        p[rng.choice(n_i, k, replace=False)] = 1 / k
        out[f"fixos_{k}"] = np.tile(p, (n_u, 1))
    for k in (100, 10):
        p = np.zeros(n_i)
        p[np.argsort(-pop)[:k]] = 1 / k
        out[f"populares_{k}"] = np.tile(p, (n_u, 1))
    top = np.argsort(-mat, axis=1)[:, :10]
    p = np.zeros((n_u, n_i))
    np.put_along_axis(p, top, 0.1, axis=1)
    out["oraculo_10"] = p
    return out


def validacao(mat, pop, n, reps, eps, corte, seed):
    rng = np.random.default_rng(seed)
    n_u, n_i = mat.shape
    pb = (1 - eps) * pop / pop.sum() + eps / n_i               # π_β(a), igual para todos os usuários
    pes = alvos(mat, pop, rng)
    verdade = {k: float((p * mat).sum(axis=1).mean()) for k, p in pes.items()}
    res = {k: {e: [] for e in ("IPS", "SNIPS", f"IPS-{corte}", "DM", "DR",
                                "IPS (π_β est.)", "SNIPS (π_β est.)", "DR (π_β est.)")} | {"ESS": []}
           for k in pes}
    for _ in range(reps):
        u = rng.integers(n_u, size=n)
        a = rng.choice(n_i, size=n, p=pb)
        r = mat[u, a]
        cnt = np.bincount(a, minlength=n_i).astype(float)
        pb_est = (cnt + 1) / (cnt.sum() + n_i)                 # contagens do log, suavização de Laplace
        soma = np.bincount(a, weights=r, minlength=n_i)
        q = np.where(cnt > 0, soma / np.maximum(cnt, 1), r.mean())   # q̂(a): média do item no log (não personalizada)
        for k, p in pes.items():
            pe_a = p[u, a]
            w, w_est = pesos(pe_a, pb[a]), pesos(pe_a, pb_est[a])
            v_q = (p @ q)[u]                                    # Σ_a π_e(a|u) q̂(a), por usuário
            d = res[k]
            d["IPS"].append(ips(w, r)); d["SNIPS"].append(snips(w, r))
            d[f"IPS-{corte}"].append(ips_corte(w, r, corte)); d["DM"].append(dm(v_q))
            d["DR"].append(dr(w, r, q[a], v_q))
            d["IPS (π_β est.)"].append(ips(w_est, r)); d["SNIPS (π_β est.)"].append(snips(w_est, r))
            d["DR (π_β est.)"].append(dr(w_est, r, q[a], v_q))
            d["ESS"].append(ess(w))
    resumo = {}
    for k, d in res.items():
        resumo[k] = {"verdadeiro": verdade[k], "ESS_medio": float(np.mean(d["ESS"])),
                     "suporte_pi_e": int((pes[k] > 0).sum(axis=1).mean())}
        for e, v in d.items():
            if e == "ESS":
                continue
            v = np.array(v)
            resumo[k][e] = {"media": float(np.nanmean(v)), "vies": float(np.nanmean(v) - verdade[k]),
                            "rmse": float(np.sqrt(np.nanmean((v - verdade[k]) ** 2)))}
    return resumo, pb


def ordem_preservada(resumo, est):
    """Fração dos pares de políticas-alvo cuja ordem pela média do estimador é a mesma da verdadeira."""
    ks = list(resumo)
    ok = tot = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            dv = resumo[ks[i]]["verdadeiro"] - resumo[ks[j]]["verdadeiro"]
            de = resumo[ks[i]][est]["media"] - resumo[ks[j]][est]["media"]
            ok += np.sign(dv) == np.sign(de)
            tot += 1
    return ok / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--validacao", action="store_true")
    p.add_argument("--n", type=int, nargs="+", default=[100_000, 1_000_000], help="tamanhos do log")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--eps", type=float, default=0.05, help="mistura uniforme na política de registro")
    p.add_argument("--corte", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/ope")
    a = p.parse_args()
    if not a.validacao:
        raise SystemExit("por enquanto só --validacao; a aplicação às políticas treinadas exige os checkpoints")
    os.makedirs(a.out, exist_ok=True)
    mat, users, items = matriz_avaliacao(a.data)
    pop = popularidade_externa(a.data, users, items)
    saida = {"config": vars(a), "itens_sem_visualizacao_externa": int((pop == 0).sum()), "por_n": {}}
    linhas = [f"Matriz {mat.shape[0]} x {mat.shape[1]}; itens sem visualização fora da avaliação: "
              f"{saida['itens_sem_visualizacao_externa']}; π_β = {1 - a.eps:.2f}·popularidade + {a.eps:.2f}·uniforme"]
    for n in a.n:
        resumo, pb = validacao(mat, pop, n, a.reps, a.eps, a.corte, a.seed)
        ests = [e for e in next(iter(resumo.values())) if isinstance(next(iter(resumo.values()))[e], dict)]
        saida["por_n"][str(n)] = {"resumo": resumo, "ordem_preservada": {e: ordem_preservada(resumo, e) for e in ests},
                                  "pi_b_min": float(pb.min()), "pi_b_max": float(pb.max())}
        linhas += ["", f"== Log com n = {n:,} passos, {a.reps} repetições (viés / RMSE contra o valor verdadeiro)",
                   f"{'alvo':13s} {'suporte':>7s} {'verdadeiro':>10s} {'ESS':>10s}" + "".join(f" {e:>22s}" for e in ests)]
        for k, d in resumo.items():
            linhas.append(f"{k:13s} {d['suporte_pi_e']:>7d} {d['verdadeiro']:>10.3f} {d['ESS_medio']:>10.0f}" +
                          "".join(f" {d[e]['vies']:>+10.3f} / {d[e]['rmse']:<9.3f}" for e in ests))
        linhas.append("Ordem dos alvos preservada (fração de pares): " +
                      "; ".join(f"{e} {v:.2f}" for e, v in saida["por_n"][str(n)]["ordem_preservada"].items()))
    json.dump(saida, open(os.path.join(a.out, "ope_validacao.json"), "w"), indent=1, ensure_ascii=False)
    txt = "\n".join(linhas)
    open(os.path.join(a.out, "ope_validacao.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
