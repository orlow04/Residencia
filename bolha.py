"""Bolha por trajetória sob NX_10, relativa ao consumo real do próprio usuário.

Definição (aprovada em 07/11/2026; ver docs/experimentos_semana8.md, item 3 do planejamento):
  - variedade V: número de categorias distintas na união das categorias dos 10 itens
    (a "variety" de Michiels et al., RecSys 2023, adaptada a itens com 1 a 4 categorias);
  - equilíbrio D: Gini-Simpson 1 − Σ q_c², em que cada item conta 1/k para cada uma das suas k categorias;
  - referência do usuário: média de V e D sobre as janelas de 10 itens consecutivos do histórico do usuário
    no big_matrix (dado de treino), em ordem de timestamp, sem repetir itens (mantém a primeira visualização),
    porque o NX_10 também não repete; a versão com repetições sai como sensibilidade;
  - índice de bolha: B_V = V_u − V e B_D = D_u − D. B > 0: trajetória mais estreita que o consumo do usuário.

Entradas:
  - trajetórias do traj_log.patch (roda_trajetorias.sh): <ORIG>_traj_s<semente>.csv, só as que têm o .ok;
  - política aleatória NX_10: gerada aqui (usuário uniforme, 10 itens distintos uniformes, como na S3).
Agregação como no resto do TCC: média sobre as 200 épocas em cada semente; média ± dp entre sementes.

Uso (de dorl-workspace/DORL-codes, onde estão os dados e as trajetórias):
  python ../../bolha.py --data environments/KuaiRec/data --traj trajetorias --out ../../results/bolha
Só precisa de numpy e pandas. A referência dos usuários fica em cache (<out>/ref_usuario.csv).
"""
import argparse
import ast
import glob
import os
import re

import numpy as np
import pandas as pd

NOMES = {"MBPO": "MBPO", "MOPO": "MOPO", "DORL": "DORL", "DORLconst": "DORL-const"}
ORDEM = ["MBPO", "MOPO", "DORL", "DORL-const", "aleatória"]


def matrizes_categorias(feats, n_cat):
    """W[i, c] = 1/k se o item i tem a categoria c (k categorias); P[i, c] = presença."""
    P = np.zeros((len(feats), n_cat), dtype=bool)
    for i, f in enumerate(feats):
        P[i, f] = True
    W = P / P.sum(1, keepdims=True)
    return W, P


def metricas(itens, W, P):
    """itens: (n, L) IDs originais. Devolve V (variedade) e D (Gini-Simpson) de cada linha."""
    q = W[itens].sum(1) / itens.shape[1]
    return P[itens].any(1).sum(1), 1 - (q ** 2).sum(1)


def janelas_usuario(seq, W, P, L):
    """Média de V e D sobre todas as janelas de L itens consecutivos de seq (somas acumuladas)."""
    if len(seq) < L:
        return np.nan, np.nan, 0
    cw = np.vstack([np.zeros(W.shape[1]), np.cumsum(W[seq], 0)])
    cp = np.vstack([np.zeros(P.shape[1], dtype=int), np.cumsum(P[seq], 0)])
    q = (cw[L:] - cw[:-L]) / L
    v = ((cp[L:] - cp[:-L]) > 0).sum(1)
    return v.mean(), (1 - (q ** 2).sum(1)).mean(), len(v)


def referencia(data, users, W, P, L):
    b = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp"])
    b = b[b["user_id"].isin(users)].sort_values(["user_id", "timestamp"], kind="mergesort")
    linhas = []
    for u, g in b.groupby("user_id", sort=True):
        seq = g["video_id"].to_numpy()
        uniq = g.drop_duplicates("video_id")["video_id"].to_numpy()
        v, d, nj = janelas_usuario(uniq, W, P, L)
        vb, db, _ = janelas_usuario(seq, W, P, L)
        linhas.append({"user_id": u, "n_interacoes": len(seq), "n_itens_distintos": len(uniq), "n_janelas": nj,
                       "V_u": v, "D_u": d, "V_u_com_repeticao": vb, "D_u_com_repeticao": db})
    ref = pd.DataFrame(linhas)
    faltam = set(users) - set(ref["user_id"])
    if faltam:
        raise SystemExit(f"{len(faltam)} usuários do small_matrix sem histórico no big_matrix")
    ref.insert(0, "user", np.searchsorted(users, ref["user_id"]))     # codificação do LabelEncoder do ambiente
    return ref.sort_values("user").reset_index(drop=True)


def le_trajetorias(path, L):
    df = pd.read_csv(path, usecols=["epoch", "protocolo", "traj", "t", "user", "item_raw"])
    df = df[df["protocolo"] == "NX_10"].sort_values(["epoch", "traj", "t"])
    n = df.groupby(["epoch", "traj"]).size()
    if (n != L).any():
        raise SystemExit(f"{path}: {int((n != L).sum())} trajetórias NX_10 com comprimento != {L}")
    ep = df["epoch"].to_numpy()[::L]
    return ep, df["user"].to_numpy()[::L], df["item_raw"].to_numpy().reshape(-1, L)


def aleatoria(rng, small_items, n_users, epochs, n_traj, L):
    n = epochs * n_traj
    users = rng.integers(0, n_users, n)
    itens = np.stack([rng.choice(small_items, L, replace=False) for _ in range(n)])
    return np.repeat(np.arange(1, epochs + 1), n_traj), users, itens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--traj", required=True, help="pasta com <ORIG>_traj_s<semente>.csv")
    p.add_argument("--out", default="results/bolha")
    p.add_argument("--L", type=int, default=10, help="comprimento do NX_10 e das janelas do usuário")
    p.add_argument("--seeds_aleatoria", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n_traj", type=int, default=100)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cat = pd.read_csv(os.path.join(a.data, "item_categories.csv"))
    assert np.array_equal(cat["video_id"].to_numpy(), np.arange(len(cat))), "o DORL indexa pela posição da linha"
    feats = cat["feat"].map(ast.literal_eval).to_list()
    W, P = matrizes_categorias(feats, max(map(max, feats)) + 1)
    small = pd.read_csv(os.path.join(a.data, "small_matrix.csv"), usecols=["user_id", "video_id"])
    users, small_items = np.unique(small["user_id"]), np.unique(small["video_id"])

    cache = os.path.join(a.out, "ref_usuario.csv")
    if os.path.isfile(cache):
        ref = pd.read_csv(cache)
        print(f"referência dos usuários lida de {cache}")
    else:
        ref = referencia(a.data, users, W, P, a.L)
        ref.to_csv(cache, index=False)
        print(f"referência dos usuários gravada em {cache}")
    assert len(ref) == len(users) and (ref["user"].to_numpy() == np.arange(len(users))).all()
    Vu, Du = ref["V_u"].to_numpy(), ref["D_u"].to_numpy()
    Vr, Dr = ref["V_u_com_repeticao"].to_numpy(), ref["D_u_com_repeticao"].to_numpy()

    execucoes = []
    for f in sorted(glob.glob(os.path.join(a.traj, "*_traj_s*.csv"))):
        m = re.fullmatch(r"(.+)_traj_s(\d+)\.csv", os.path.basename(f))
        if not m or m.group(1) not in NOMES:
            continue
        if not os.path.isfile(f + ".ok"):
            print(f"ignorado (sem .ok, incompleto ou não conferido): {f}")
            continue
        execucoes.append((NOMES[m.group(1)], int(m.group(2)), *le_trajetorias(f, a.L)))
    for s in a.seeds_aleatoria:
        execucoes.append(("aleatória", s, *aleatoria(np.random.default_rng(s), small_items, len(users),
                                                     a.epochs, a.n_traj, a.L)))

    por_epoca, por_semente = [], []
    for metodo, semente, ep, u, itens in execucoes:
        V, D = metricas(itens, W, P)
        t = pd.DataFrame({"epoch": ep, "V": V, "D": D, "B_V": Vu[u] - V, "B_D": Du[u] - D,
                          "B_V_pos": Vu[u] - V > 0, "B_D_pos": Du[u] - D > 0,
                          "B_V_com_repeticao": Vr[u] - V, "B_D_com_repeticao": Dr[u] - D, "V_u": Vu[u], "D_u": Du[u]})
        e = t.groupby("epoch").mean()
        e.insert(0, "semente", semente); e.insert(0, "metodo", metodo)
        por_epoca.append(e.reset_index())
        por_semente.append({"metodo": metodo, "semente": semente, "n_epocas": len(e), "n_traj": len(t),
                            **e.drop(columns=["metodo", "semente"]).mean().to_dict()})
        print(f"{metodo:10s} s{semente}: {len(e)} épocas, {len(t)} trajetórias")

    por_epoca = pd.concat(por_epoca)
    por_semente = pd.DataFrame(por_semente)
    por_epoca.to_csv(os.path.join(a.out, "bolha_por_epoca.csv"), index=False)
    por_semente.to_csv(os.path.join(a.out, "bolha_por_semente.csv"), index=False)

    cols = ["V", "V_u", "B_V", "B_V_pos", "D", "D_u", "B_D", "B_D_pos", "B_V_com_repeticao", "B_D_com_repeticao"]
    g = por_semente.groupby("metodo")[cols]
    resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
    resumo["n"] = por_semente.groupby("metodo").size()
    resumo = resumo.reindex([m for m in ORDEM if m in resumo.index])
    resumo.to_csv(os.path.join(a.out, "bolha_resumo.csv"))

    fmt = lambda m, c, nd: f"{resumo.loc[m, ('media', c)]:.{nd}f} ± {resumo.loc[m, ('dp', c)]:.{nd}f}"
    linhas = [f"Bolha por trajetória, NX_10 (L = {a.L}); média entre sementes ± dp entre sementes",
              "B = referência do usuário − trajetória; B > 0: mais estreita que o consumo real do usuário",
              f"Referência: janelas de {a.L} itens consecutivos do big_matrix, sem repetição de itens "
              f"(média entre usuários: V_u {np.nanmean(Vu):.2f}, D_u {np.nanmean(Du):.3f})", "",
              f"{'Método':11s} {'n':>2s}  {'V':>13s} {'B_V':>14s} {'B_V>0':>13s}  {'D':>15s} {'B_D':>16s} {'B_D>0':>13s}"]
    for m in resumo.index:
        linhas.append(f"{m:11s} {int(resumo.loc[m, 'n'].iloc[0]):>2d}  {fmt(m, 'V', 2):>13s} {fmt(m, 'B_V', 2):>14s} "
                      f"{fmt(m, 'B_V_pos', 3):>13s}  {fmt(m, 'D', 3):>15s} {fmt(m, 'B_D', 3):>16s} "
                      f"{fmt(m, 'B_D_pos', 3):>13s}")
    linhas += ["", "Sensibilidade: referência com repetições de itens no histórico", f"{'Método':11s}  {'B_V':>14s} {'B_D':>16s}"]
    linhas += [f"{m:11s}  {fmt(m, 'B_V_com_repeticao', 2):>14s} {fmt(m, 'B_D_com_repeticao', 3):>16s}"
               for m in resumo.index]
    texto = "\n".join(linhas)
    print("\n" + texto)
    with open(os.path.join(a.out, "bolha_saida.txt"), "w") as f:
        f.write(texto + "\n")
    print(f"\ngravado em {a.out}: ref_usuario.csv, bolha_por_epoca.csv, bolha_por_semente.csv, "
          "bolha_resumo.csv, bolha_saida.txt")


if __name__ == "__main__":
    main()
