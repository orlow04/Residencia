"""alem_da_acuracia.py: cada critério por época, comparado com a definição calculada item a item, em dados sintéticos.
Não há código do DORL a copiar (o artigo não usa esses critérios), exceto a MCD, já testada em test_random.py.

Uso (da raiz do repositório):  python tests/test_alem.py "$PWD/alem_da_acuracia.py"
"""
import importlib.util, itertools, math, sys
import numpy as np

spec = importlib.util.spec_from_file_location("alem", sys.argv[1])
alem = importlib.util.module_from_spec(spec); spec.loader.exec_module(alem)

rng = np.random.default_rng(1)
n_u, n_i, n_cat, L, n_traj = 30, 80, 9, 10, 25
raw = np.sort(rng.choice(500, n_i, replace=False))                 # IDs originais dos itens do "small_matrix"
feats = [list(rng.choice(n_cat, int(rng.integers(1, 4)), replace=False)) for _ in range(500)]
W, P = alem.matrizes_categorias(feats, n_cat)
mat = rng.gamma(1.0, 1.0, (n_u, n_i)).clip(max=5)
pop = rng.random(n_i)
topcat = np.zeros((n_u, n_cat), dtype=bool)
for u in range(n_u):
    topcat[u, rng.choice(n_cat, 3, replace=False)] = True
ctx = {"mat": mat, "media_u": mat.mean(1), "raw": raw, "W": W, "P": P, "Pcat": P[raw], "topcat": topcat,
       "is_dom": rng.random(n_i) < 0.6, "nov": -np.log2(pop / pop.sum()), "top_pop": rng.random(n_i) < 0.2}

ep = np.repeat([1, 2, 3], n_traj)
u = rng.integers(0, n_u, len(ep))
it = np.stack([rng.choice(n_i, L, replace=False) for _ in ep])
res = alem.por_epoca(ep, u, it, ctx).set_index("epoch")


def gini_def(x):  # média das diferenças absolutas entre pares ÷ (2 × média)
    x = np.asarray(x, float)
    return np.abs(x[:, None] - x[None, :]).mean() / (2 * x.mean())


for e in (1, 2, 3):
    s = np.flatnonzero(ep == e)
    itens = [(int(u[j]), int(i)) for j in s for i in it[j]]
    rel = {(uu, i): mat[uu, i] > mat[uu].mean() for uu, i in itens}
    esperado = {
        "R_each": np.mean([mat[uu, i] for uu, i in itens]),
        "MCD": np.mean([ctx["is_dom"][i] for _, i in itens]),
        "V": np.mean([len(set(itertools.chain(*(feats[raw[i]] for i in it[j])))) for j in s]),
        "cobertura": len({i for _, i in itens}) / len(itens),
        "Gini": gini_def([sum(1 for _, i in itens if i == k) for k in range(n_i)]),
        "novidade": np.mean([-math.log2(pop[i] / pop.sum()) for _, i in itens]),
        "seren_pop": np.mean([rel[(uu, i)] and not ctx["top_pop"][i] for uu, i in itens]),
        "seren_cat": np.mean([rel[(uu, i)] and not any(topcat[uu, c] for c in feats[raw[i]]) for uu, i in itens]),
        "relevante": np.mean([rel[(uu, i)] for uu, i in itens]),
        "personalizacao": 1 - np.mean([len(set(it[a]) & set(it[b])) / len(set(it[a]) | set(it[b]))
                                       for a, b in itertools.combinations(s, 2)]),
    }
    for k, v in esperado.items():
        assert np.isclose(res.loc[e, k], v), (e, k, res.loc[e, k], v)

assert np.isclose(alem.gini(np.ones(7)), 0) and np.isclose(alem.gini([0] * 9 + [5]), 0.9)
print("OK: R_each, MCD, V, cobertura, Gini, novidade, serendipidade (pop e cat), relevância e personalização "
      "idênticos à definição item a item")
