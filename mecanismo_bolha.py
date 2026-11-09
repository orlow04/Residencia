"""De onde vem a diversidade das trajetórias NX_10? Duas hipóteses, medidas nas trajetórias gravadas.

1. Regra de saída: a política aprendeu a não repetir categoria. Mede-se a fração dos passos (t ≥ 1) que teriam
   disparado a regra se ela estivesse ativa no NX_10 (KuaiEnv._determine_whether_to_leave, com o recorte original da
   janela e com a janela corrigida), e a fração das trajetórias que não a disparariam nenhuma vez.
2. Categorias dominantes: a política evita as 10 categorias dominantes (MCD baixa) e, com isso, vai para a cauda de
   categorias. Mede-se o V esperado dado o número k de itens dominantes na trajetória: para cada k = 0..L, a média do
   V de trajetórias sorteadas com k itens dominantes e L − k não dominantes, uniformes e sem repetição (aleatória
   condicionada). Se o V esperado ≈ V observado, a diversidade é explicada pela fuga das categorias dominantes.
   Referência do usuário: fração de itens dominantes nas janelas de L itens do big_matrix (sem repetição).

Uso (de dorl-workspace/DORL-codes):
  python ../../mecanismo_bolha.py --data environments/KuaiRec/data \
      --domination environments/KuaiRec/env/feature_domination.pickle --traj trajetorias --out ../../results/mecanismo
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alem_da_acuracia import contexto  # noqa: E402
from bolha import aleatoria, execucoes_gravadas, metricas, ordem  # noqa: E402


def janela(t, N, correta):
    """Posições comparadas no passo t pela regra de saída (lista com t itens, como em KuaiEnv.sequence_action)."""
    if t == 0:
        return range(0)
    if correta:
        return range(max(0, t - N), t)
    ini = t - N if t - N >= 0 else max(0, 2 * t - N)                 # fatiamento [t − N:t] numa lista de t itens
    return range(ini, t)


def disparos(itens_raw, P, N, correta):
    """(n, L) booleano: o passo t teria disparado a regra de saída (leave_threshold = 0)."""
    pres = P[itens_raw]                                               # (n, L, n_cat)
    n, L = itens_raw.shape
    out = np.zeros((n, L), dtype=bool)
    for t in range(1, L):
        for j in janela(t, N, correta):
            out[:, t] |= (pres[:, t] & pres[:, j]).any(-1)
    return out


def v_condicional(rng, items, is_dom, W, P, L, n_amostras):
    """V médio de trajetórias com k itens dominantes (k = 0..L), sorteados sem repetição."""
    dom, nao = items[is_dom], items[~is_dom]
    vk = np.full(L + 1, np.nan)
    for k in range(L + 1):
        if k > len(dom) or L - k > len(nao):
            continue
        tr = np.stack([np.concatenate([rng.choice(dom, k, replace=False), rng.choice(nao, L - k, replace=False)])
                       for _ in range(n_amostras)])
        vk[k] = metricas(tr, W, P)[0].mean()
    return vk


def dominantes_usuario(data, users, is_dom_raw, L):
    b = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp"])
    b = b[b["user_id"].isin(users)].sort_values(["user_id", "timestamp"], kind="mergesort")
    b = b.drop_duplicates(["user_id", "video_id"])
    fr = []
    for _, g in b.groupby("user_id", sort=True):
        d = is_dom_raw[g["video_id"].to_numpy()].astype(float)
        if len(d) >= L:
            c = np.concatenate([[0], np.cumsum(d)])
            fr.append(((c[L:] - c[:-L]) / L).mean())
    return float(np.mean(fr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--domination", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--out", default="results/mecanismo")
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--N", type=int, default=4, help="N da regra de saída usado no contrafactual")
    p.add_argument("--n_amostras", type=int, default=20000)
    p.add_argument("--seeds_aleatoria", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n_traj", type=int, default=100)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ctx = contexto(a.data, a.domination, 0.8, 100, 3)
    W, P, items, is_dom = ctx["W"], ctx["P"], ctx["items"], ctx["is_dom"]
    # item dominante pelo ID original (inclui itens fora do small_matrix, para as janelas do usuário)
    is_dom_raw = np.array([any(c + 1 in ctx["dominadas"] for c in np.flatnonzero(P[i])) for i in range(len(P))])
    assert (is_dom_raw[items] == is_dom).all()
    vk = v_condicional(np.random.default_rng(0), items, is_dom, W, P, a.L, a.n_amostras)
    fr_u = dominantes_usuario(a.data, ctx["users"], is_dom_raw, a.L)

    execs = execucoes_gravadas(a.traj, a.L)
    for s in a.seeds_aleatoria:
        execs.append(("aleatória", s, *aleatoria(np.random.default_rng(s), items, len(ctx["users"]),
                                                 a.epochs, a.n_traj, a.L)))
    sem = []
    for metodo, semente, ep, u, it in execs:
        k = is_dom_raw[it].sum(1)
        V = metricas(it, W, P)[0]
        d_orig, d_corr = disparos(it, P, a.N, False), disparos(it, P, a.N, True)
        sem.append({"metodo": metodo, "semente": semente, "frac_dom": k.mean() / a.L, "V": V.mean(),
                    "V_esperado_k": np.nanmean(vk[k]), "disparo_passo": d_orig[:, 1:].mean(),
                    "sem_disparo": (~d_orig.any(1)).mean(), "disparo_passo_corrigida": d_corr[:, 1:].mean(),
                    "sem_disparo_corrigida": (~d_corr.any(1)).mean()})
        print(f"{metodo:10s} s{semente}")
    sem = pd.DataFrame(sem)
    sem.to_csv(os.path.join(a.out, "mecanismo_por_semente.csv"), index=False)
    cols = ["frac_dom", "V", "V_esperado_k", "disparo_passo", "sem_disparo", "disparo_passo_corrigida",
            "sem_disparo_corrigida"]
    g = sem.groupby("metodo")[cols]
    resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
    resumo["n"] = sem.groupby("metodo").size()
    resumo = resumo.reindex(ordem(resumo.index))
    resumo.to_csv(os.path.join(a.out, "mecanismo_resumo.csv"))

    txt = [f"Mecanismo da diversidade, NX_10 (L = {a.L}); média entre sementes ± dp entre sementes",
           f"Usuário (janelas de {a.L} itens do big_matrix, sem repetição): fração de itens dominantes "
           f"{fr_u:.3f}",
           "V esperado dado k itens dominantes (aleatória condicionada): " +
           ", ".join(f"k={k}: {v:.2f}" for k, v in enumerate(vk)),
           f"Regra de saída contrafactual com N = {a.N}: fração dos passos t ≥ 1 que a disparariam; fração das "
           "trajetórias sem nenhum disparo (recorte original | janela corrigida)", ""]
    nd = {"frac_dom": 3, "V": 2, "V_esperado_k": 2, "disparo_passo": 3, "sem_disparo": 3,
          "disparo_passo_corrigida": 3, "sem_disparo_corrigida": 3}
    rot = {"frac_dom": "frac. dom.", "V": "V", "V_esperado_k": "V esperado(k)", "disparo_passo": "disparo/passo",
           "sem_disparo": "sem disparo", "disparo_passo_corrigida": "disp./passo corr.",
           "sem_disparo_corrigida": "sem disp. corr."}
    txt.append(f"{'Método':11s} {'n':>2s} " + " ".join(f"{rot[c]:>17s}" for c in cols))
    for m in resumo.index:
        txt.append(f"{m:11s} {int(resumo.loc[m, 'n'].iloc[0]):>2d} " + " ".join(
            f"{resumo.loc[m, ('media', c)]:.{nd[c]}f} ± {resumo.loc[m, ('dp', c)]:.{nd[c]}f}".rjust(17) for c in cols))
    texto = "\n".join(txt)
    print("\n" + texto)
    open(os.path.join(a.out, "mecanismo_saida.txt"), "w").write(texto + "\n")


if __name__ == "__main__":
    main()
