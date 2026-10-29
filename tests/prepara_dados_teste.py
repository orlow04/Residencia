"""Prepara os dados sintéticos usados por test_evmat.py e test_random.py.

Parte de uma pasta gerada por make_synthetic_kuairec.py e cria outra com:
  - small_matrix.csv com casos-limite: uma linha duplicada, uma célula ausente, um NaN e um valor > 5;
  - big_matrix.csv e item_categories.csv copiados;
  - matPre.pickle falso (50 x 100) e dom.pickle falso (categorias dominantes, codificação +1 do DORL).

Uso: python tests/prepara_dados_teste.py <pasta_sintetica> <pasta_saida>
"""
import ast
import collections
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)

sm = pd.read_csv(os.path.join(src, "small_matrix.csv"))
sm = pd.concat([sm, sm.iloc[[5]]])                       # duplicata
sm = sm.drop(sm.index[10])                               # célula ausente
sm.iloc[20, sm.columns.get_loc("watch_ratio")] = np.nan  # NaN
sm.iloc[30, sm.columns.get_loc("watch_ratio")] = 7.0     # > 5
sm.to_csv(os.path.join(out, "small_matrix.csv"), index=False)

for f in ("big_matrix.csv", "item_categories.csv"):
    shutil.copy(os.path.join(src, f), os.path.join(out, f))

n_users, n_items = sm["user_id"].nunique(), sm["video_id"].nunique()
rng = np.random.default_rng(0)
pickle.dump(rng.normal(0, 0.05, (n_users, n_items)), open(os.path.join(out, "matPre.pickle"), "wb"))

cat = pd.read_csv(os.path.join(src, "item_categories.csv"))
c = collections.Counter(x + 1 for f in cat.feat.map(ast.literal_eval) for x in f)
tot = sum(c.values())
dom = {"feat": sorted([(np.int64(k), v / tot * 0.1) for k, v in c.items()], key=lambda x: -x[1])}
pickle.dump(dom, open(os.path.join(out, "dom.pickle"), "wb"))
print(f"dados de teste em {out}: small {sm.shape}, matPre ({n_users}, {n_items})")
