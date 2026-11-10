"""sessoes_reais.marca(): elegibilidade (cadeia contígua de N elos), repetição de categoria nos N anteriores e fim de
sessão, comparados com a definição calculada linha a linha, em dados sintéticos.
Uso (da raiz do repositório):  python tests/test_sessoes.py "$PWD/sessoes_reais.py"
"""
import importlib.util, sys
import numpy as np

spec = importlib.util.spec_from_file_location("ses", sys.argv[1])
ses = importlib.util.module_from_spec(spec); spec.loader.exec_module(ses)
rng = np.random.default_rng(5)
n_cat, n_items = 6, 40
feats = [list(rng.choice(n_cat, int(rng.integers(1, 3)), replace=False)) for _ in range(n_items)]
_, P = ses.matrizes_categorias(feats, n_cat)
user = np.sort(rng.integers(0, 8, 3000))
play = rng.uniform(1, 20, len(user))
ts = np.zeros(len(user))
for i in range(1, len(user)):   # intervalos ora contíguos (fim + 0..4 s), ora com buracos, ora longos
    ts[i] = ts[i - 1] + play[i - 1] + rng.choice([rng.uniform(0, 4), rng.uniform(10, 300), rng.uniform(2000, 9000)], p=[.7, .2, .1])
item = rng.integers(0, n_items, len(user))
for N in (1, 3, 4):
    for folga, G in ((2, 30), (5, 10)):
        eleg, rep, fim = ses.marca(user, ts, play, item, P, N, folga, G)
        for t in range(len(user)):
            e = t >= N and all(user[t - j] == user[t] for j in range(1, N + 1)) and \
                all(ts[t - j + 1] - (ts[t - j] + play[t - j]) <= folga for j in range(1, N + 1))
            assert eleg[t] == e, (N, t)
            r = e and any(set(feats[item[t]]) & set(feats[item[t - j]]) for j in range(1, N + 1))
            assert rep[t] == r, (N, t)
            f = t == len(user) - 1 or user[t + 1] != user[t] or ts[t + 1] - ts[t] > G * 60
            assert fim[t] == f, (N, t)
print("OK: elegibilidade, repetição e fim de sessão idênticos à definição linha a linha")
