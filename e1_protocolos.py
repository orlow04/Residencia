"""E1 — inversão por protocolo: as mesmas políticas, ordenadas por métrica e por protocolo de avaliação.

Usa a mesma leitura e agregação do table2_from_log.py (média sobre as épocas de cada execução).
Depois agrega ENTRE SEMENTES (média e desvio padrão) e ordena os métodos em cada protocolo e métrica.

Protocolos (código do DORL, collector_set.py):
  FB     itens podem se repetir; regra de saída ativa
  NX_0   itens não se repetem;   regra de saída ativa   (Tabela 2 do artigo)
  NX_10  itens não se repetem;   trajetória forçada a 10 rodadas (Length fixo; R_tra = 10 x R_each)

Uso (coloque este arquivo na mesma pasta do table2_from_log.py):
  python e1_protocolos.py \
    "MBPO=saved_models/KuaiEnv-v0/A2C_with_emb/logs/[[]MBPO_s[0-9][]]_*.log" \
    "MOPO=saved_models/KuaiEnv-v0/A2C_with_emb/logs/[[]MOPO_s[0-9][]]_*.log" \
    "DORL=saved_models/KuaiEnv-v0/A2C_with_emb/logs/[[]DORL_s[0-9][]]_*.log" \
    "UCB=saved_models/KuaiEnv-v0/LinUCB/logs/[[]UCB[]]_*.log" \
    "eps-greedy=saved_models/KuaiEnv-v0/EpsilonGreedy/logs/[[]epsilon-greedy[]]_*.log" \
    --out e1
  ([[] e []] são colchetes literais no glob.)
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table2_from_log import read_log, summarize  # noqa: E402

PROTOS = {"FB": "", "NX_0": "NX_0_", "NX_10": "NX_10_"}
METRICS = ["R_tra", "R_each", "Length", "MCD"]
LOWER_IS_BETTER = {"MCD"}

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help='"NOME=glob dos logs" (um glob por método; cada log é uma semente)')
p.add_argument("--out", default="e1")
p.add_argument("--aleatoria", default=None, help="aleatoria.json do politica_aleatoria.py (linha de referência)")
a = p.parse_args()

# 1. Valor por semente: média sobre as épocas (igual à Tabela 2)
rows = []
for spec in a.runs:
    name, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"nenhum log para {name}: {pattern}")
    for f in files:
        df = read_log(f)
        df = df[df.index >= 0]
        for proto, prefix in PROTOS.items():
            for metric, (mean, _) in summarize(df, prefix).items():
                rows.append({"metodo": name, "log": os.path.basename(f), "protocolo": proto,
                             "metrica": metric, "valor": mean})
if a.aleatoria:  # mesma agregação: média sobre as épocas em cada semente
    import json
    for seed, protos in json.load(open(a.aleatoria))["por_semente"].items():
        for proto, metrics in protos.items():
            for metric, (mean, _) in metrics.items():
                rows.append({"metodo": "aleatória", "log": f"aleatoria_s{seed}", "protocolo": proto,
                             "metrica": metric, "valor": mean})
seeds = pd.DataFrame(rows)
seeds.to_csv(f"{a.out}_por_semente.csv", index=False)

# 2. Agregação entre sementes
agg = (seeds.groupby(["protocolo", "metrica", "metodo"])["valor"]
       .agg(media="mean", dp="std", n="count").reset_index())
agg.to_csv(f"{a.out}_resumo.csv", index=False)
methods = list(dict.fromkeys(s.split("=", 1)[0] for s in a.runs)) + (["aleatória"] if a.aleatoria else [])
get = agg.set_index(["protocolo", "metrica", "metodo"])


def valid(proto, metric):
    v = agg[(agg.protocolo == proto) & (agg.metrica == metric)]["media"]
    return len(v) == len(methods) and v.max() - v.min() > 1e-9   # NX_10/Length é constante


print("== Média entre sementes ± dp entre sementes (n = número de sementes). MCD: menor é melhor.")
for proto in PROTOS:
    print(f"\n{proto}")
    print(f"  {'método':12s} {'n':>2s}  " + "  ".join(f"{m:>16s}" for m in METRICS))
    for meth in methods:
        cells = []
        for m in METRICS:
            r = get.loc[(proto, m, meth)]
            cells.append(f"{r.media:8.3f} ± {r.dp:5.3f}" if r.n > 1 else f"{r.media:8.3f}  (1 sem.)")
        print(f"  {meth:12s} {int(get.loc[(proto, METRICS[0], meth)].n):2d}  " + "  ".join(f"{c:>16s}" for c in cells))

# 3. Ordenação
print("\n== Posição de cada método (1 = melhor)")
cols = [(pr, m) for pr in PROTOS for m in METRICS if valid(pr, m)]
rank = pd.DataFrame(index=methods)
for pr, m in cols:
    s = get.loc[(pr, m)]["media"].reindex(methods)
    rank[f"{pr}:{m}"] = s.rank(ascending=m in LOWER_IS_BETTER, method="min").astype(int)
print(rank.to_string())
rank.to_csv(f"{a.out}_posicoes.csv")


def better(proto, metric, x, y):
    """+1 se x é melhor que y, -1 se pior; e se a diferença supera o maior dp entre sementes."""
    rx, ry = get.loc[(proto, metric, x)], get.loc[(proto, metric, y)]
    d = rx.media - ry.media
    if metric in LOWER_IS_BETTER:
        d = -d
    sds = [s for s in (rx.dp, ry.dp) if not np.isnan(s)]
    robust = None if not sds else abs(d) > max(sds)
    return np.sign(d), robust


def tag(res):
    """Inversão robusta: ao menos uma vitória e uma derrota que superam o dp entre sementes."""
    if any(r is None for _, r in res.values()):
        return "[sem dp: 1 semente nos dois]"
    ok = any(r for s, r in res.values() if s > 0) and any(r for s, r in res.values() if s < 0)
    return "[supera o dp entre sementes]" if ok else "[dentro da variação entre sementes]"


lines_m, lines_p = [], []
for x, y in itertools.combinations(methods, 2):
    for pr in PROTOS:                                         # entre métricas, mesmo protocolo
        res = {m: better(pr, m, x, y) for m in METRICS if valid(pr, m)}
        wins = [m for m, (s, _) in res.items() if s > 0]
        loss = [m for m, (s, _) in res.items() if s < 0]
        if wins and loss:
            lines_m.append(f"  {pr:5s} {x} × {y}: {x} melhor em {', '.join(wins)}; {y} melhor em {', '.join(loss)} "
                           + tag(res))
    for m in METRICS:                                         # entre protocolos, mesma métrica
        res = {pr: better(pr, m, x, y) for pr in PROTOS if valid(pr, m)}
        wins = [pr for pr, (s, _) in res.items() if s > 0]
        loss = [pr for pr, (s, _) in res.items() if s < 0]
        if wins and loss:
            lines_p.append(f"  {m:6s} {x} × {y}: {x} melhor em {', '.join(wins)}; {y} melhor em {', '.join(loss)} "
                           + tag(res))

print("\n== Inversões entre métricas (mesmo protocolo)")
print("\n".join(lines_m) or "  nenhuma")
print("\n== Inversões entre protocolos (mesma métrica)")
print("\n".join(lines_p) or "  nenhuma")
print(f"\nArquivos: {a.out}_por_semente.csv, {a.out}_resumo.csv, {a.out}_posicoes.csv")
