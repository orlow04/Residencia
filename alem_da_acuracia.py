"""Critérios de avaliação além dos do artigo, nas mesmas trajetórias NX_10: a ordem dos métodos depende do critério?

Critérios (por época de 100 trajetórias de 10 itens; média sobre as épocas em cada semente; média ± dp entre sementes):
  do artigo     R_each   recompensa de avaliação média por item (KuaiEnv.load_mat: watch_ratio limitado a 5)
                MCD      fração dos itens com alguma categoria dominante (evaluator.get_feat_dominate_dict)
  da bolha      V        categorias distintas por trajetória (bolha.py)
  exposição     cobertura   itens distintos ÷ itens recomendados na época (1 = nenhum item repetido entre usuários)
                Gini        Gini da exposição dos 3.327 itens do small_matrix na época (0 = exposição igual)
  novidade      novidade    média de −log2 da popularidade do item no big_matrix (Kaminskas & Bridge, 2016)
  serendipidade relevante e inesperado (Ge et al., RecSys 2010): relevante = recompensa de avaliação acima da média
                do usuário no small_matrix; inesperado = fora dos K itens mais populares no big_matrix (seren_pop) ou
                sem nenhuma categoria entre as C mais consumidas pelo usuário no big_matrix (seren_cat)
  personalização 1 − Jaccard médio entre as listas de pares de trajetórias da mesma época
Inversões: como no e1_protocolos.py, um par de métodos inverte quando cada um é melhor que o outro em algum critério e
as duas diferenças superam o maior dp entre sementes dos dois.

Uso (de dorl-workspace/DORL-codes):
  python ../../alem_da_acuracia.py --data environments/KuaiRec/data \
      --domination environments/KuaiRec/env/feature_domination.pickle --traj trajetorias --out ../../results/alem
Precisa de numpy, pandas e scipy (o venv do DORL tem). Lê só os CSVs com .ok (como o bolha.py).
"""
import argparse
import ast
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bolha import aleatoria, execucoes_gravadas, matrizes_categorias, metricas, ordem  # noqa: E402

# critério → True se maior é melhor (MCD: menor é melhor, como no artigo; Gini: menor é exposição mais igual)
CRITERIOS = {"R_each": True, "MCD": False, "V": True, "cobertura": True, "Gini": False, "novidade": True,
             "seren_pop": True, "seren_cat": True, "personalizacao": True}


def gini(x):
    """Gini de um vetor não negativo (0 = todos iguais; perto de 1 = tudo num só)."""
    x = np.sort(np.asarray(x, dtype=float))
    n, s = len(x), x.sum()
    return 0.0 if s == 0 else float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * s))


def personalizacao(itens):
    """itens: (n, L) índices; 1 − Jaccard médio entre os pares de linhas (listas sem repetição)."""
    n, L = itens.shape
    X = np.zeros((n, int(itens.max()) + 1), dtype=np.float32)
    X[np.repeat(np.arange(n), L), itens.ravel()] = 1
    inter = X @ X.T
    iu = np.triu_indices(n, 1)
    return 1 - float((inter[iu] / (2 * L - inter[iu])).mean())


def por_epoca(ep, u, it, ctx):
    """Métricas de cada época. u: usuários codificados; it: (n, L) índices codificados no small_matrix."""
    rel = ctx["mat"][u[:, None], it] > ctx["media_u"][u][:, None]
    linhas = []
    for e in np.unique(ep):
        s = ep == e
        ue, ie, re_ = u[s], it[s], rel[s]
        V, _ = metricas(ctx["raw"][ie], ctx["W"], ctx["P"])
        inesp_cat = ~(ctx["Pcat"][ie] & ctx["topcat"][ue][:, None, :]).any(-1)
        linhas.append({"epoch": e,
                       "R_each": ctx["mat"][ue[:, None], ie].mean(),
                       "MCD": ctx["is_dom"][ie].mean(),
                       "V": V.mean(),
                       "cobertura": len(np.unique(ie)) / ie.size,
                       "Gini": gini(np.bincount(ie.ravel(), minlength=ctx["mat"].shape[1])),
                       "novidade": ctx["nov"][ie].mean(),
                       "seren_pop": (re_ & ~ctx["top_pop"][ie]).mean(),
                       "seren_cat": (re_ & inesp_cat).mean(),
                       "relevante": re_.mean(),
                       "personalizacao": personalizacao(ie)})
    return pd.DataFrame(linhas)


def contexto(data, dom_path, top_rate, k_pop, c_cat):
    small = pd.read_csv(os.path.join(data, "small_matrix.csv"), usecols=["user_id", "video_id", "watch_ratio"])
    users, ui = np.unique(small["user_id"].to_numpy(), return_inverse=True)
    items, ii = np.unique(small["video_id"].to_numpy(), return_inverse=True)
    mat = np.zeros((len(users), len(items)))                      # KuaiEnv.load_mat (sem pares duplicados no KuaiRec)
    np.add.at(mat, (ui, ii), small["watch_ratio"].clip(upper=5).to_numpy())
    mat[~np.isfinite(mat)] = small["watch_ratio"].clip(upper=5).mean()

    cat = pd.read_csv(os.path.join(data, "item_categories.csv"))
    assert np.array_equal(cat["video_id"].to_numpy(), np.arange(len(cat))), "o DORL indexa pela posição da linha"
    feats = cat["feat"].map(ast.literal_eval).to_list()
    n_cat = max(map(max, feats)) + 1
    W, P = matrizes_categorias(feats, n_cat)

    dom = pickle.load(open(dom_path, "rb"))["feat"]               # mesmo laço de get_feat_dominate_dict
    v = np.array([float(x[1]) for x in dom])
    cum = (v / v.sum()).cumsum()
    ind = max(int(np.sum(np.cumsum(cum > top_rate) == 0)), 1)
    dominadas = {int(x[0]) for x in dom[:ind]}                    # codificação +1 do load_category
    is_dom = np.array([any(c + 1 in dominadas for c in feats[i]) for i in items])

    big = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id"])
    cont = np.bincount(big["video_id"].to_numpy(), minlength=len(feats)).astype(float)
    p = (cont + 1) / (cont.sum() + len(cont))                     # suavização +1 (itens nunca vistos)
    top_pop = np.zeros(len(items), dtype=bool)
    top_pop[np.argsort(-cont[items], kind="stable")[:k_pop]] = True

    b = big[big["user_id"].isin(users)]
    C = csr_matrix((np.ones(len(b)), (np.searchsorted(users, b["user_id"].to_numpy()), b["video_id"].to_numpy())),
                   shape=(len(users), len(feats)))                # visualizações por usuário e item
    peso = np.asarray(C @ W)                                      # cada visualização conta 1/k para cada categoria
    topcat = np.zeros((len(users), n_cat), dtype=bool)
    topcat[np.arange(len(users))[:, None], np.argsort(-peso, axis=1, kind="stable")[:, :c_cat]] = True

    return {"mat": mat, "media_u": mat.mean(1), "users": users, "items": items, "raw": items, "W": W, "P": P,
            "Pcat": P[items], "topcat": topcat, "is_dom": is_dom, "nov": -np.log2(p[items]), "top_pop": top_pop,
            "dominadas": sorted(dominadas)}


def inversoes(resumo, metodos):
    linhas = []
    for a, b in itertools.combinations(metodos, 2):
        melhor = {a: [], b: []}
        for c, maior in CRITERIOS.items():
            d = resumo.loc[a, ("media", c)] - resumo.loc[b, ("media", c)]
            dp = np.nanmax([resumo.loc[a, ("dp", c)], resumo.loc[b, ("dp", c)]])
            if abs(d) > dp:
                melhor[a if (d > 0) == maior else b].append(c)
        if melhor[a] and melhor[b]:
            linhas.append(f"  {a} × {b}: {a} melhor em {', '.join(melhor[a])}; {b} melhor em {', '.join(melhor[b])}")
    return linhas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--domination", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--out", default="results/alem")
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--top_rate", type=float, default=0.8)
    p.add_argument("--k_pop", type=int, default=100, help="itens populares considerados esperados (seren_pop)")
    p.add_argument("--c_cat", type=int, default=3, help="categorias do usuário consideradas esperadas (seren_cat)")
    p.add_argument("--seeds_aleatoria", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n_traj", type=int, default=100)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ctx = contexto(a.data, a.domination, a.top_rate, a.k_pop, a.c_cat)
    execs = execucoes_gravadas(a.traj, a.L)
    for s in a.seeds_aleatoria:
        execs.append(("aleatória", s, *aleatoria(np.random.default_rng(s), ctx["items"], len(ctx["users"]),
                                                 a.epochs, a.n_traj, a.L)))
    eps, sem = [], []
    for metodo, semente, ep, u, itens_raw in execs:
        it = np.searchsorted(ctx["items"], itens_raw)
        assert (ctx["items"][it] == itens_raw).all(), "item fora do small_matrix"
        e = por_epoca(ep, u, it, ctx)
        e.insert(0, "semente", semente); e.insert(0, "metodo", metodo)
        eps.append(e)
        sem.append({"metodo": metodo, "semente": semente, "n_epocas": len(e),
                    **e.drop(columns=["metodo", "semente", "epoch"]).mean().to_dict()})
        print(f"{metodo:10s} s{semente}: {len(e)} épocas")
    eps, sem = pd.concat(eps), pd.DataFrame(sem)
    eps.to_csv(os.path.join(a.out, "alem_por_epoca.csv"), index=False)
    sem.to_csv(os.path.join(a.out, "alem_por_semente.csv"), index=False)

    cols = list(CRITERIOS) + ["relevante"]
    g = sem.groupby("metodo")[cols]
    resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
    resumo["n"] = sem.groupby("metodo").size()
    metodos = ordem(resumo.index)
    resumo = resumo.reindex(metodos)
    resumo.to_csv(os.path.join(a.out, "alem_resumo.csv"))

    nd = {"R_each": 3, "MCD": 3, "V": 2, "cobertura": 3, "Gini": 3, "novidade": 2, "seren_pop": 3, "seren_cat": 3,
          "relevante": 3, "personalizacao": 4}
    txt = [f"Critérios nas trajetórias NX_10 (L = {a.L}); média entre sementes ± dp entre sementes",
           f"seren_pop: inesperado = fora dos {a.k_pop} itens mais populares; seren_cat: nenhuma categoria entre as "
           f"{a.c_cat} mais consumidas pelo usuário; relevante = recompensa acima da média do usuário",
           "Melhor: maior em todos, exceto MCD e Gini (menor)", ""]
    txt.append(f"{'Método':11s} {'n':>2s} " + " ".join(f"{c:>16s}" for c in cols))
    for m in metodos:
        txt.append(f"{m:11s} {int(resumo.loc[m, 'n'].iloc[0]):>2d} " + " ".join(
            f"{resumo.loc[m, ('media', c)]:.{nd[c]}f} ± {resumo.loc[m, ('dp', c)]:.{nd[c]}f}".rjust(16) for c in cols))
    pos = pd.DataFrame({c: resumo[("media", c)].rank(ascending=not maior, method="min").astype(int)
                        for c, maior in CRITERIOS.items()})
    txt += ["", "Posição de cada método (1 = melhor)", pos.to_string(), "",
            "Inversões entre critérios (as duas diferenças superam o maior dp entre sementes)"]
    txt += inversoes(resumo, metodos) or ["  nenhuma"]
    texto = "\n".join(txt)
    print("\n" + texto)
    open(os.path.join(a.out, "alem_saida.txt"), "w").write(texto + "\n")
    print(f"\ngravado em {a.out}: alem_por_epoca.csv, alem_por_semente.csv, alem_resumo.csv, alem_saida.txt")


if __name__ == "__main__":
    main()
