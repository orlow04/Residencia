"""mecanismo_bolha.py e fb_rotacao.py, em dados sintéticos.
  - disparos(): igual a uma cópia literal de KuaiEnv._determine_whether_to_leave (recorte original) e, com a janela
    corrigida, à regra sobre os min(t, N) anteriores;
  - v_condicional(): igual a uma reimplementação independente (Monte Carlo; tolerância 0,05);
  - fb_rotacao.medidas(): casos montados à mão.
Uso (da raiz do repositório):  python tests/test_mecanismo.py "$PWD/mecanismo_bolha.py" "$PWD/fb_rotacao.py"
"""
import importlib.util, itertools, sys
from collections import Counter
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


mec, fb = load(sys.argv[1], "mec"), load(sys.argv[2], "fb")


class Env:  # cópia literal de KuaiEnv._determine_whether_to_leave (commit 818c9bd)
    def __init__(self, seq, lfs, N): self.sequence_action, self.list_feat_small, self.num_leave_compute, self.leave_threshold = seq, lfs, N, 0
    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))
        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True
        return False


rng = np.random.default_rng(3)
n_items, n_cat, L = 60, 8, 10
feats = [list(rng.choice(n_cat, int(rng.integers(1, 4)), replace=False)) for _ in range(n_items)]
W, P = sys.modules["bolha"].matrizes_categorias(feats, n_cat)      # o mecanismo_bolha.py já importou o bolha.py
itens = np.stack([rng.choice(n_items, L, replace=False) for _ in range(400)])
for N in range(1, 10):
    d_orig, d_corr = mec.disparos(itens, P, N, False), mec.disparos(itens, P, N, True)
    for r, seq in enumerate(itens):
        for t in range(L):
            assert d_orig[r, t] == Env(list(seq[:t]), feats, N)._determine_whether_to_leave(t, seq[t]), (N, r, t)
            janela = seq[max(0, t - N):t] if t else []
            assert d_corr[r, t] == any(set(feats[seq[t]]) & set(feats[j]) for j in janela), (N, r, t)

items = np.arange(n_items)
is_dom = rng.random(n_items) < 0.5
vk = mec.v_condicional(np.random.default_rng(0), items, is_dom, W, P, L, 4000)
r2 = np.random.default_rng(9)
dom, nao = items[is_dom], items[~is_dom]
for k in range(L + 1):
    if k > len(dom) or L - k > len(nao):
        assert np.isnan(vk[k]); continue
    v = np.mean([len(set(itertools.chain(*(feats[i] for i in list(r2.choice(dom, k, replace=False)) +
                                           list(r2.choice(nao, L - k, replace=False))))))
                 for _ in range(4000)])
    assert abs(v - vk[k]) < 0.05, (k, v, vk[k])

P2 = np.eye(5, dtype=bool)                 # item i tem só a categoria i
m = fb.medidas([0, 1, 2, 0, 1, 2, 0], P2, 10)
assert m["ciclo_puro"] and m["p_ciclo"] == 3 and m["periodo"] == 3 and m["frac_periodo"] == 1
assert m["itens_distintos"] == 3 and m["categorias_distintas"] == 3 and np.isclose(m["repeticao"], 1 - 3 / 7)
m = fb.medidas([4, 4, 4], P2, 10)
assert m["ciclo_puro"] and m["p_ciclo"] == 1
m = fb.medidas([0, 1, 0, 2], P2, 10)
assert not m["ciclo_puro"] and m["periodo"] == 2 and m["frac_periodo"] == 0.5
m = fb.medidas([0, 1, 2, 3, 4], P2, 10)
assert not m["ciclo_puro"] and m["repeticao"] == 0
print("OK: regra contrafactual idêntica à cópia de _determine_whether_to_leave (e à janela corrigida); V condicional "
      "igual à reimplementação; medidas do FB corretas nos casos montados")
