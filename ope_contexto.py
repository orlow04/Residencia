"""Validação semi-sintética do OPE com política de registro dependente do contexto (complementa ope.py --validacao).

No caso real, π_β depende do histórico e é desconhecida; o erro ao estimá-la entra no viés. Aqui o contexto é
x = (usuário u, último item l), e a política de registro verdadeira é a transição real do big_matrix:
  π_β(a | l) = (1 − ε)·(C[l, a] + α·pop(a)) / (C[l]·1 + α) + ε / n_itens
com C = contagens de pares consecutivos (l → a) nas sessões dos usuários fora da avaliação (itens do small_matrix, sem
duplicatas, sessão encerrada após 30 min de intervalo), pop = popularidade normalizada, α = 10 e ε = 0,05. O último
item é sorteado pela popularidade e o usuário ao acaso; a recompensa é a do small_matrix (KuaiEnv.load_mat).

π_β usada nos pesos, três versões:
  verdadeira   a política acima (limite ideal)
  contagens    (c[l, a] + β·marg(a)) / (c[l]·1 + β), contada no próprio log, β = 10 (bem especificada, mas esparsa)
  marginal     marg(a) = frequência do item no log, ignorando l (mal especificada)
Políticas-alvo: uniforme; 30 itens fixos ao acaso; oráculo top-10 por usuário; os 10 próximos itens mais frequentes
depois de l (segue o registro); e 100 itens fixos restritos aos que não compartilham categoria com l (imita o RL, que
aprende a não repetir categoria).

Uso (de dorl-workspace/DORL-codes):
  python ../../ope_contexto.py --data environments/KuaiRec/data --out ../../results/ope
"""
import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ope import dm, dr, ess, ips, matriz_avaliacao, pesos, snips  # noqa: E402

SESSAO = 1800


def transicoes(data, users, items):
    """Contagens C[l, a] de pares consecutivos na mesma sessão e popularidade, fora dos usuários de avaliação."""
    b = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    b = b.drop_duplicates()
    b = b[~b["user_id"].isin(set(users)) & b["video_id"].isin(set(items))]
    b = b.sort_values(["user_id", "timestamp"], kind="mergesort")
    idx = np.searchsorted(items, b["video_id"].to_numpy())
    u, t = b["user_id"].to_numpy(), b["timestamp"].to_numpy()
    par = (u[1:] == u[:-1]) & (np.diff(t) <= SESSAO)
    n = len(items)
    C = np.bincount(idx[:-1][par] * n + idx[1:][par], minlength=n * n).reshape(n, n).astype(float)
    pop = np.bincount(idx, minlength=n).astype(float)
    return C, pop / pop.sum(), int(par.sum())


def categorias(data, items):
    cat = pd.read_csv(os.path.join(data, "item_categories.csv"))
    feats = cat["feat"].map(ast.literal_eval).to_list()             # o DORL indexa pela posição da linha
    return [set(feats[i]) for i in items]


def alvos(mat, C, pop, cats, rng):
    """{nome: (tipo, matriz)}: tipo "x" (igual em todo contexto, vetor), "u" (usuário × item) ou "l" (último × item)."""
    n_u, n_i = mat.shape
    out = {"uniforme": ("x", np.full(n_i, 1 / n_i))}
    p = np.zeros(n_i)
    p[rng.choice(n_i, 30, replace=False)] = 1 / 30
    out["fixos_30"] = ("x", p)
    top = np.argsort(-mat, axis=1)[:, :10]
    p = np.zeros((n_u, n_i))
    np.put_along_axis(p, top, 0.1, axis=1)
    out["oraculo_10"] = ("u", p)
    seg = np.argsort(-(C + 1e-9 * pop), axis=1)[:, :10]            # empates (linhas vazias) pela popularidade
    p = np.zeros((n_i, n_i))
    np.put_along_axis(p, seg, 0.1, axis=1)
    out["segue_registro_10"] = ("l", p)
    base = rng.choice(n_i, 100, replace=False)
    p = np.zeros((n_i, n_i))
    for l in range(n_i):
        ok = [a for a in base if not (cats[a] & cats[l])] or list(base)
        p[l, ok] = 1 / len(ok)
    out["cat_diferente_100"] = ("l", p)
    return out


def valor_verdadeiro(tipo, p, mat, pop):
    m = mat.mean(axis=0)                                             # recompensa média do item sobre os usuários
    if tipo == "x":
        return float(p @ m)
    if tipo == "u":
        return float((p * mat).sum(axis=1).mean())
    return float(pop @ (p @ m))                                      # l ~ pop


def amostra_log(rng, n, pb, pop, n_u):
    n_i = len(pop)
    l = rng.choice(n_i, size=n, p=pop)
    u = rng.integers(n_u, size=n)
    a = np.empty(n, dtype=np.int64)
    ordem = np.argsort(l, kind="stable")
    ls, ini = np.unique(l[ordem], return_index=True)
    fim = np.append(ini[1:], n)
    for x, i, j in zip(ls, ini, fim):
        a[ordem[i:j]] = rng.choice(n_i, size=j - i, p=pb[x])
    return u, l, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[1_000_000, 10_000_000])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--alfa", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ope")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    mat, users, items = matriz_avaliacao(a.data)
    n_u, n_i = mat.shape
    C, pop, n_pares = transicoes(a.data, users, items)
    pb = (1 - a.eps) * (C + a.alfa * pop) / (C.sum(axis=1, keepdims=True) + a.alfa) + a.eps / n_i
    cats = categorias(a.data, items)
    pes = alvos(mat, C, pop, cats, rng)
    verdade = {k: valor_verdadeiro(t, p, mat, pop) for k, (t, p) in pes.items()}
    versoes = ("verdadeira", "contagens", "marginal")
    ests = [f"{e} ({v})" for v in versoes for e in ("IPS", "SNIPS", "DR")]
    saida = {"config": vars(a), "pares_consecutivos": n_pares, "linhas_sem_transicao": int((C.sum(axis=1) == 0).sum()),
             "verdadeiro": verdade, "por_n": {}}
    linhas = [f"Pares consecutivos no big_matrix (fora da avaliação, itens do small, mesma sessão): {n_pares:,}; "
              f"itens sem transição de saída: {saida['linhas_sem_transicao']}",
              f"π_β verdadeira = {1 - a.eps:.2f}·transição (α = {a.alfa:g}) + {a.eps:.2f}·uniforme; "
              f"contagens com β = {a.beta:g}"]
    for n in a.n:
        res = {k: {e: [] for e in ests + ["DM"]} | {"ESS": []} for k in pes}
        for _ in range(a.reps):
            u, l, act = amostra_log(rng, n, pb, pop, n_u)
            r = mat[u, act]
            marg = np.bincount(act, minlength=n_i).astype(float)
            c_la = np.bincount(l * n_i + act, minlength=n_i * n_i).reshape(n_i, n_i).astype(float)
            marg_n = (marg + 1) / (marg.sum() + n_i)
            pi = {"verdadeira": pb[l, act], "marginal": marg_n[act],
                  "contagens": (c_la[l, act] + a.beta * marg_n[act]) / (c_la.sum(axis=1)[l] + a.beta)}
            cnt = marg
            q = np.where(cnt > 0, np.bincount(act, weights=r, minlength=n_i) / np.maximum(cnt, 1), r.mean())
            for k, (t, p) in pes.items():
                pe_a = p[act] if t == "x" else (p[u, act] if t == "u" else p[l, act])
                vq = np.full(n, p @ q) if t == "x" else (p @ q)[u if t == "u" else l]
                d = res[k]
                d["DM"].append(dm(vq))
                for v in versoes:
                    w = pesos(pe_a, pi[v])
                    d[f"IPS ({v})"].append(ips(w, r))
                    d[f"SNIPS ({v})"].append(snips(w, r))
                    d[f"DR ({v})"].append(dr(w, r, q[act], vq))
                    if v == "verdadeira":
                        d["ESS"].append(ess(w))
        resumo = {}
        for k, d in res.items():
            resumo[k] = {"verdadeiro": verdade[k], "ESS_medio": float(np.mean(d["ESS"]))}
            for e in ests + ["DM"]:
                v = np.array(d[e])
                resumo[k][e] = {"media": float(v.mean()), "vies": float(v.mean() - verdade[k]),
                                "rmse": float(np.sqrt(np.mean((v - verdade[k]) ** 2)))}
        saida["por_n"][str(n)] = resumo
        linhas += ["", f"== Log com n = {n:,} passos, {a.reps} repetições (viés / RMSE)",
                   f"{'alvo':18s} {'verdadeiro':>10s} {'ESS':>9s}" + "".join(f" {e:>22s}" for e in ests + ["DM"])]
        for k, d in resumo.items():
            linhas.append(f"{k:18s} {d['verdadeiro']:>10.3f} {d['ESS_medio']:>9.0f}" +
                          "".join(f" {d[e]['vies']:>+10.3f} / {d[e]['rmse']:<9.3f}" for e in ests + ["DM"]))
    json.dump(saida, open(os.path.join(a.out, "ope_contexto.json"), "w"), indent=1, ensure_ascii=False)
    txt = "\n".join(linhas)
    open(os.path.join(a.out, "ope_contexto.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
