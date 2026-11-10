"""e6_saida_prob.patch: sem E6_P_REP, KuaiEnv._determine_whether_to_leave decide como o original e não consome números
aleatórios; com E6_P_REP/E6_P_BASE, sai com essas probabilidades conforme o item repita ou não categoria da janela
(a mesma repetição que a regra original usa), de forma reprodutível com random.seed.

Uso (de dorl-workspace/DORL-codes):
  python ../../tests/test_e6.py <KuaiEnv.py original> <KuaiEnv.py com o patch>
"""
import importlib.util, os, random, sys
import numpy as np

ORIG, PATCHED = sys.argv[1:3]
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])
P_REP, P_BASE = 0.3, 0.05


def load(path, name, env):
    for k in ("E6_P_REP", "E6_P_BASE", "A5_FIX"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k in env:
        os.environ.pop(k)
    return mod.KuaiEnv._determine_whether_to_leave


class Estado:
    def __init__(self, seq, feats, N):
        self.sequence_action, self.list_feat_small, self.num_leave_compute, self.leave_threshold = seq, feats, N, 0


orig = load(ORIG, "k_orig", {})
sem = load(PATCHED, "k_sem", {})
com = load(PATCHED, "k_com", {"E6_P_REP": str(P_REP), "E6_P_BASE": str(P_BASE)})

rng = np.random.default_rng(0)
casos = []
for _ in range(4000):
    n_items, n_cat = int(rng.integers(5, 40)), int(rng.integers(2, 10))
    feats = [list(rng.choice(n_cat, int(rng.integers(1, min(3, n_cat) + 1)), replace=False)) for _ in range(n_items)]
    seq = list(rng.integers(0, n_items, 30))
    N = int(rng.integers(1, 9))
    for t in range(1, 30):
        casos.append((Estado(seq[:t], feats, N), t, int(rng.integers(0, n_items))))

random.seed(1); antes = random.getstate()
rep = [orig(st, t, a) for st, t, a in casos]
assert [sem(st, t, a) for st, t, a in casos] == rep, "sem as variáveis a decisão mudou"
assert random.getstate() == antes, "sem as variáveis o patch consumiu números aleatórios"

random.seed(7); d1 = np.array([com(st, t, a) for st, t, a in casos])
random.seed(7); d2 = np.array([com(st, t, a) for st, t, a in casos])
assert (d1 == d2).all(), "não reprodutível com a mesma semente"
rep = np.array(rep)                                   # no original, sair = repetir categoria da janela
f_rep, f_base = d1[rep].mean(), d1[~rep].mean()
se = lambda p, n: 4 * np.sqrt(p * (1 - p) / n)        # 4 erros padrão
assert abs(f_rep - P_REP) < se(P_REP, rep.sum()) and abs(f_base - P_BASE) < se(P_BASE, (~rep).sum()), (f_rep, f_base)
print(f"OK: {len(casos):,} decisões; sem as variáveis idêntico ao original e sem consumir números aleatórios; com "
      f"E6_P_REP={P_REP}/E6_P_BASE={P_BASE}: saída {f_rep:.3f} quando repete ({rep.sum():,}) e {f_base:.3f} quando não "
      f"({(~rep).sum():,}); reprodutível com random.seed")
