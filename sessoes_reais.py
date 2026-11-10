"""A regra de saída do simulador tem apoio no comportamento real? Teste nas sessões do big_matrix.

O usuário simulado (KuaiEnv._determine_whether_to_leave) assiste ao item e sai se ele repete alguma categoria dos
N itens anteriores; senão continua (até 30 passos). Nos dados reais pergunta-se: a sessão termina mais vezes logo
depois de um vídeo que repete categoria dos N anteriores?

Problema dos dados: o big_matrix só tem os 10.728 vídeos do KuaiRec, não o feed completo. Duas linhas consecutivas de
um usuário têm, em mediana, 108 s de intervalo para 7 s assistidos: há vídeos fora dos dados no meio. Por isso o teste
usa só cadeias contíguas: a linha t entra se ela e as N anteriores formam uma sequência em que cada vídeo começa até
FOLGA segundos depois do fim do anterior (timestamp + play_duration). Nessas cadeias, os N anteriores dos dados são os
N anteriores do feed.

  repete(t)   alguma categoria do vídeo t aparece nos N anteriores da cadeia (a regra do simulador, janela completa)
  fim(t)      o próximo vídeo do usuário começa mais de G minutos depois (ou não há próximo): a sessão termina após t
Comparação: P(fim | repete) contra P(fim | não repete), no total e dentro de cada usuário (média das diferenças
por usuário, o que controla a heterogeneidade entre usuários). O simulador supõe 1 contra 0.

Uso (da raiz do repositório; roda no Mac em poucos minutos):
  python sessoes_reais.py --data dorl-workspace/DORL-codes/environments/KuaiRec/data --out results/sessoes
"""
import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bolha import matrizes_categorias  # noqa: E402


def marca(user, ts, play, item, P, N, folga, G):
    """Arrays por linha (ordenadas por usuário e timestamp): elegível, repete, fim."""
    mesmo = np.r_[False, user[1:] == user[:-1]]
    contig = mesmo & np.r_[False, ts[1:] - (ts[:-1] + play[:-1]) <= folga]       # elo (t − 1 → t) sem vídeo faltando
    prox = np.r_[user[:-1] == user[1:], False]
    fim = ~prox | np.r_[ts[1:] - ts[:-1] > G * 60, True]
    # cadeia: os elos t−N+1..t são todos contíguos (c[i] = nº de elos não contíguos entre as posições 0..i−1)
    c = np.r_[0, np.cumsum(~contig)]
    eleg = np.zeros(len(user), dtype=bool)
    eleg[N:] = (c[N + 1:] - c[1:len(user) - N + 1]) == 0
    repete = np.zeros(len(user), dtype=bool)
    for j in range(1, N + 1):
        repete[j:] |= (P[item[j:]] & P[item[:-j]]).any(1)
    return eleg, repete & eleg, fim


def resumo(user, eleg, repete, fim):
    d = pd.DataFrame({"u": user[eleg], "r": repete[eleg], "f": fim[eleg]})
    tot = d.groupby("r")["f"].agg(["mean", "size"])
    pu = d.groupby(["u", "r"])["f"].mean().unstack()
    pu = pu.dropna()
    return {"linhas_elegiveis": int(eleg.sum()), "frac_repete": float(d["r"].mean()),
            "P_fim_repete": float(tot.loc[True, "mean"]) if True in tot.index else np.nan,
            "P_fim_nao_repete": float(tot.loc[False, "mean"]) if False in tot.index else np.nan,
            "n_repete": int(tot.loc[True, "size"]) if True in tot.index else 0,
            "n_nao_repete": int(tot.loc[False, "size"]) if False in tot.index else 0,
            "usuarios_com_os_dois": int(len(pu)),
            "dif_media_por_usuario": float((pu[True] - pu[False]).mean()),
            "dif_dp_por_usuario": float((pu[True] - pu[False]).std(ddof=1)),
            "frac_usuarios_dif_positiva": float(((pu[True] - pu[False]) > 0).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/sessoes")
    p.add_argument("--N", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--G", type=float, nargs="+", default=[10, 30, 60], help="minutos sem atividade que encerram a sessão")
    p.add_argument("--folga", type=float, nargs="+", default=[2, 5], help="segundos entre o fim de um vídeo e o início do próximo")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cat = pd.read_csv(os.path.join(a.data, "item_categories.csv"))
    assert np.array_equal(cat["video_id"].to_numpy(), np.arange(len(cat))), "o DORL indexa pela posição da linha"
    feats = cat["feat"].map(ast.literal_eval).to_list()
    _, P = matrizes_categorias(feats, max(map(max, feats)) + 1)
    b = pd.read_csv(os.path.join(a.data, "big_matrix.csv"),
                    usecols=["user_id", "video_id", "timestamp", "play_duration"]).dropna()
    b = b.sort_values(["user_id", "timestamp"], kind="mergesort")
    user, item = b["user_id"].to_numpy(), b["video_id"].to_numpy().astype(int)
    ts, play = b["timestamp"].to_numpy(), b["play_duration"].to_numpy() / 1000

    linhas = []
    for folga in a.folga:
        for N in a.N:
            for G in a.G:
                eleg, rep, fim = marca(user, ts, play, item, P, N, folga, G)
                linhas.append({"folga_s": folga, "N": N, "G_min": G, **resumo(user, eleg, rep, fim)})
                r = linhas[-1]
                print(f"folga {folga:g}s N={N} G={G:g}min: {r['linhas_elegiveis']:,} linhas; repete {r['frac_repete']:.3f}; "
                      f"P(fim|repete) {r['P_fim_repete']:.3f} × P(fim|não) {r['P_fim_nao_repete']:.3f}; "
                      f"dif. por usuário {r['dif_media_por_usuario']:+.4f} (n={r['usuarios_com_os_dois']})")
    t = pd.DataFrame(linhas)
    t.to_csv(os.path.join(a.out, "sessoes_resumo.csv"), index=False)
    cols = ["folga_s", "N", "G_min", "linhas_elegiveis", "frac_repete", "P_fim_repete", "P_fim_nao_repete",
            "dif_media_por_usuario", "dif_dp_por_usuario", "frac_usuarios_dif_positiva", "usuarios_com_os_dois"]
    texto = ("Regra de saída × sessões reais do big_matrix (só cadeias contíguas; ver docstring)\n"
             "O simulador supõe P(fim | repete) = 1 e P(fim | não repete) = 0\n\n" +
             t[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    open(os.path.join(a.out, "sessoes_saida.txt"), "w").write(texto + "\n")
    print("\n" + texto)


if __name__ == "__main__":
    main()
