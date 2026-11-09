"""Protocolo FB (itens podem se repetir): como o MBPO sobrevive ~28 passos de 30?

Hipótese (semana 8): a política alterna poucos itens de categorias disjuntas, num ciclo maior que a janela da regra
de saída, que assim nunca dispara. Por trajetória FB, mede-se:
  Length; itens distintos; categorias distintas; repetição = 1 − itens distintos ÷ Length;
  período: para p = 1..P, fração dos passos t ≥ p com item[t] = item[t − p]; reporta-se o p de maior fração;
  ciclo puro: a trajetória inteira é a repetição de um bloco de p itens distintos (p ≤ P).
Média sobre as trajetórias (100 por época, então igual à média das épocas) em cada semente; média ± dp entre sementes.

Uso (de dorl-workspace/DORL-codes):
  python ../../fb_rotacao.py --data environments/KuaiRec/data --traj trajetorias --out ../../results/fb
Lê só os CSVs com .ok. Só precisa de numpy e pandas.
"""
import argparse
import ast
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bolha import matrizes_categorias, nome, ordem  # noqa: E402


def medidas(seq, P, p_max):
    """Medidas de uma trajetória (seq: IDs originais na ordem dos passos)."""
    seq = np.asarray(seq)
    n = len(seq)
    per = np.array([(seq[p:] == seq[:-p]).mean() if n > p else np.nan for p in range(1, p_max + 1)])
    melhor = int(np.nanargmax(per)) + 1 if np.isfinite(per).any() else 0
    ciclo = 0
    for p in range(1, p_max + 1):
        if n > p and len(set(seq[:p])) == p and (seq[p:] == seq[:-p]).all():
            ciclo = p
            break
    return {"Length": n, "itens_distintos": len(np.unique(seq)), "categorias_distintas": int(P[seq].any(0).sum()),
            "repeticao": 1 - len(np.unique(seq)) / n, "periodo": melhor,
            "frac_periodo": per[melhor - 1] if melhor else np.nan, "ciclo_puro": ciclo > 0, "p_ciclo": ciclo}


def le_fb(path):
    df = pd.read_csv(path, usecols=["epoch", "protocolo", "traj", "t", "item_raw"])
    df = df[df["protocolo"] == "FB"].sort_values(["epoch", "traj", "t"])
    return df.groupby(["epoch", "traj"])["item_raw"].agg(list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--out", default="results/fb")
    p.add_argument("--p_max", type=int, default=10)
    p.add_argument("--metodos", nargs="*", default=None, help="só estes prefixos de CSV (ex.: MBPO DORL)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cat = pd.read_csv(os.path.join(a.data, "item_categories.csv"))
    feats = cat["feat"].map(ast.literal_eval).to_list()
    _, P = matrizes_categorias(feats, max(map(max, feats)) + 1)

    sem, periodos = [], []
    for f in sorted(glob.glob(os.path.join(a.traj, "*_traj_s*.csv"))):
        m = re.fullmatch(r"(.+)_traj_s(\d+)\.csv", os.path.basename(f))
        if not m or nome(m.group(1)) is None or (a.metodos and m.group(1) not in a.metodos):
            continue
        if not os.path.isfile(f + ".ok"):
            continue
        metodo, semente = nome(m.group(1)), int(m.group(2))
        t = pd.DataFrame([medidas(s, P, a.p_max) for s in le_fb(f)])    # 100 por época: média = média das épocas
        sem.append({"metodo": metodo, "semente": semente, **t.mean().to_dict()})
        periodos.append(pd.Series(t.loc[t["ciclo_puro"], "p_ciclo"].value_counts(normalize=True), name=metodo))
        print(f"{metodo:10s} s{semente}: {len(t)} trajetórias FB")
    sem = pd.DataFrame(sem)
    sem.to_csv(os.path.join(a.out, "fb_por_semente.csv"), index=False)
    cols = ["Length", "itens_distintos", "categorias_distintas", "repeticao", "periodo", "frac_periodo", "ciclo_puro"]
    g = sem.groupby("metodo")[cols]
    resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
    resumo["n"] = sem.groupby("metodo").size()
    resumo = resumo.reindex(ordem(resumo.index))
    resumo.to_csv(os.path.join(a.out, "fb_resumo.csv"))

    rot = {"Length": "Length", "itens_distintos": "itens dist.", "categorias_distintas": "categ. dist.",
           "repeticao": "repetição", "periodo": "período", "frac_periodo": "frac. período", "ciclo_puro": "ciclo puro"}
    txt = ["Protocolo FB: estrutura das trajetórias; média entre sementes ± dp entre sementes",
           f"período: p (1..{a.p_max}) com maior fração de passos em que item[t] = item[t − p]; "
           "ciclo puro: a trajetória inteira repete um bloco de p itens distintos", ""]
    txt.append(f"{'Método':11s} {'n':>2s} " + " ".join(f"{rot[c]:>15s}" for c in cols))
    for m in resumo.index:
        txt.append(f"{m:11s} {int(resumo.loc[m, 'n'].iloc[0]):>2d} " + " ".join(
            f"{resumo.loc[m, ('media', c)]:.2f} ± {resumo.loc[m, ('dp', c)]:.2f}".rjust(15) for c in cols))
    if periodos:
        # fillna(0) antes da média: um bloco ausente numa semente conta 0 nela (e não some da média)
        dist = pd.concat(periodos, axis=1).T.fillna(0).groupby(level=0).mean().reindex(ordem(resumo.index))
        dist = dist[sorted(dist.columns)]
        txt += ["", "Entre as trajetórias em ciclo puro: fração com cada tamanho de bloco p (média entre sementes)",
                dist.round(3).to_string()]
    texto = "\n".join(txt)
    print("\n" + texto)
    open(os.path.join(a.out, "fb_saida.txt"), "w").write(texto + "\n")


if __name__ == "__main__":
    main()
