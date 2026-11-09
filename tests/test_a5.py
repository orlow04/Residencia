"""a5_janela.patch: sem A5_FIX, KuaiEnv._determine_whether_to_leave decide exatamente como o original; com a variável,
a janela passa a ser os min(t, N) itens anteriores. Também confere que o original recorta a janela onde o A5 diz.

Chama o método das duas classes sobre um objeto mínimo (sem carregar dados), em sequências aleatórias de categorias.
Uso (de dorl-workspace/DORL-codes, onde ficam src/ e o venv do DORL):
  python ../../tests/test_a5.py <KuaiEnv.py original> <KuaiEnv.py com o patch>
"""
import importlib.util, itertools, os, sys
from collections import Counter
import numpy as np

ORIG, PATCHED = sys.argv[1:3]
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])


def load(path, name, a5):
    os.environ.pop("A5_FIX", None)
    if a5:
        os.environ["A5_FIX"] = "1"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    os.environ.pop("A5_FIX", None)
    return mod.KuaiEnv._determine_whether_to_leave


class Estado:  # só os atributos que _determine_whether_to_leave lê
    def __init__(self, seq, feats, N, thr):
        self.sequence_action, self.list_feat_small, self.num_leave_compute, self.leave_threshold = seq, feats, N, thr


def referencia(seq, feats, t, action, N, thr):
    """Definição pretendida: sai se alguma categoria do item atual aparece mais de thr vezes nos min(t, N) anteriores."""
    if t == 0:
        return False
    hist = Counter(itertools.chain(*(feats[x] for x in seq[max(0, t - N):t])))
    return any(hist[c] > thr for c in feats[action])


orig, sem, com = load(ORIG, "k_orig", False), load(PATCHED, "k_sem", False), load(PATCHED, "k_com", True)
rng = np.random.default_rng(0)
n_casos, n_dif, recortes = 0, 0, set()
for _ in range(3000):
    n_items, n_cat = int(rng.integers(5, 60)), int(rng.integers(2, 12))
    feats = [list(rng.choice(n_cat, int(rng.integers(1, min(4, n_cat) + 1)), replace=False)) for _ in range(n_items)]
    seq = list(rng.integers(0, n_items, 30))
    N, thr = int(rng.integers(1, 11)), int(rng.integers(0, 2))
    for t in range(30):
        st, a = Estado(seq[:t], feats, N, thr), int(rng.integers(0, n_items))   # a lista tem t itens no passo t
        o = orig(st, t, a)
        assert sem(st, t, a) == o, "sem A5_FIX a decisão mudou"
        assert com(st, t, a) == referencia(seq, feats, t, a, N, thr), "com A5_FIX a janela não é a dos min(t, N)"
        if t > 0 and len(seq[:t][t - N:t]) < min(t, N):
            recortes.add((N, t))
        n_casos += 1
        n_dif += o != com(st, t, a)

# Onde o original recorta (janela menor que min(t, N)): t < N e 2t − N > 0, ou seja, N/2 < t < N.
esperado = {(N, t) for N in range(1, 11) for t in range(1, 30) if N / 2 < t < N}
assert recortes == esperado, (sorted(recortes ^ esperado))
assert n_dif > 0
print(f"OK: {n_casos:,} decisões; sem A5_FIX idêntico ao original; com A5_FIX janela = min(t, N) anteriores "
      f"({n_dif:,} decisões mudam). Recorte do original em N/2 < t < N (com N = 4: t = 3; com N = 8: t = 5, 6, 7)")
