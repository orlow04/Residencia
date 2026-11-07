"""bolha.py: V, D e as janelas do usuário (somas acumuladas) contra a definição calculada item a item.

Uso: python tests/test_bolha.py <bolha.py>
"""
import importlib.util, sys
from collections import Counter
import numpy as np

spec = importlib.util.spec_from_file_location("bolha", sys.argv[1])
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

rng = np.random.default_rng(0)
n_cat, L = 7, 10
feats = [sorted(rng.choice(n_cat, rng.integers(1, 5), replace=False).tolist()) for _ in range(60)]
W, P = b.matrizes_categorias(feats, n_cat)


def literal(itens):
    """Definição direta: V = categorias distintas na união; D = 1 − Σ q², item com k categorias vale 1/k a cada."""
    q = Counter()
    for i in itens:
        for c in feats[i]:
            q[c] += 1 / len(feats[i]) / len(itens)
    return len({c for i in itens for c in feats[i]}), 1 - sum(x * x for x in q.values())


itens = np.stack([rng.choice(60, L, replace=False) for _ in range(200)])
V, D = b.metricas(itens, W, P)
for row, v, d in zip(itens, V, D):
    lv, ld = literal(row)
    assert v == lv and abs(d - ld) < 1e-12

for n in (L - 1, L, L + 1, 57):
    seq = rng.integers(0, 60, n)
    v, d, nj = b.janelas_usuario(seq, W, P, L)
    if n < L:
        assert np.isnan(v) and np.isnan(d) and nj == 0
        continue
    lit = [literal(seq[s:s + L]) for s in range(n - L + 1)]
    assert nj == len(lit)
    assert abs(v - np.mean([x[0] for x in lit])) < 1e-12 and abs(d - np.mean([x[1] for x in lit])) < 1e-12

# casos de referência: tudo numa categoria -> V 1, D 0; 10 categorias únicas distintas -> V 10, D 0,9
um = [[0]] * 10 + [[c] for c in range(10)]
W1, P1 = b.matrizes_categorias(um, 10)
V1, D1 = b.metricas(np.array([np.arange(10), np.arange(10, 20)]), W1, P1)
assert list(V1) == [1, 10] and np.allclose(D1, [0, 0.9])

ep, us, it = b.aleatoria(np.random.default_rng(1), np.arange(30, 90), 5, 3, 4, L)
assert it.shape == (12, L) and all(len(set(r)) == L for r in it) and it.min() >= 30 and list(ep) == [1] * 4 + [2] * 4 + [3] * 4
print("OK: V e D por trajetória e médias nas janelas do usuário idênticas à definição item a item; "
      "aleatória sem repetição de itens")
