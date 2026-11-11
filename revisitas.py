"""Revisualização de vídeos: o que os usuários reais fazem e o que as políticas fazem no protocolo FB.

O DORL não usa a repetição como métrica. Mas ela aparece em dois lugares sem ser nomeada: (1) a recompensa de
avaliação é o watch_ratio bruto limitado a 5 (KuaiEnv.load_mat), e watch_ratio > 1 é o vídeo repetido dentro da mesma
visualização (loop); (2) no FB a política pode repetir itens, e o item repetido recebe a mesma recompensa da primeira
vez (KuaiEnv.py:317). Este script mede a referência real e a compara com as políticas.

A. Revisitas no big_matrix (usuários de avaliação e todos). Remove-se as duplicatas exatas (mesmo usuário, vídeo,
   timestamp e watch_ratio). Cada nova ocorrência de um par usuário–vídeo é classificada pelo intervalo desde a
   anterior: imediata (≤ 60 s), mesma sessão (60 s a 30 min) ou outra sessão (> 30 min, a revisita propriamente dita).
   Concentração: fração das revisitas nos 10% de itens mais populares contra a fração de todas as visualizações;
   Gini das revisitas por item contra o das visualizações. Engajamento: watch_ratio da revisita ÷ o da visualização
   anterior do mesmo par (mediana). Sessões (corte de 30 min, como em sessoes_reais.py): repetição dentro da sessão
   = 1 − itens distintos ÷ tamanho, a mesma definição do fb_rotacao.py.
B. Loop dentro da visualização (small_matrix, células observadas): fração com watch_ratio > 1 e parte da recompensa
   de avaliação, min(wr, 5), que vem de wr acima de 1.
C. (opcional, --traj) Trajetórias FB das políticas: repetição; fração dos passos repetidos em itens do top 10% de
   popularidade; cobertura das épocas contando e não contando repetições. Média sobre épocas por semente; média ± dp
   entre sementes.

Uso (de dorl-workspace/DORL-codes):
  python ../../revisitas.py --data environments/KuaiRec/data --out ../../results/revisitas            # A e B (Mac)
  python ../../revisitas.py --data environments/KuaiRec/data --traj trajetorias --out ../../results/revisitas  # + C (4090)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bolha import nome, ordem  # noqa: E402

IMEDIATA, SESSAO = 60, 1800


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else np.nan


def revisitas(b, top):
    """b: big_matrix sem duplicatas exatas. Devolve o dicionário de medidas da parte A."""
    b = b.sort_values(["user_id", "video_id", "timestamp"], kind="mergesort")
    dt = b.groupby(["user_id", "video_id"])["timestamp"].diff()
    wr_ant = b.groupby(["user_id", "video_id"])["watch_ratio"].shift()
    rep = dt.notna()
    tipo = pd.cut(dt[rep], [-np.inf, IMEDIATA, SESSAO, np.inf], labels=["imediata", "mesma_sessao", "outra_sessao"])
    rev = rep & (dt > SESSAO)
    n = len(b)
    vis_item = b["video_id"].value_counts()
    rev_item = b.loc[rev, "video_id"].value_counts().reindex(vis_item.index, fill_value=0)
    razao = (b.loc[rev, "watch_ratio"] / wr_ant[rev]).replace([np.inf, -np.inf], np.nan).dropna()

    s = b.sort_values(["user_id", "timestamp"], kind="mergesort")
    nova = (s["user_id"] != s["user_id"].shift()) | (s["timestamp"].diff() > SESSAO)
    sess = s.assign(sessao=nova.cumsum()).groupby("sessao")["video_id"].agg(["size", "nunique"])
    sess = sess[sess["size"] >= 2]
    rep_sess = 1 - sess["nunique"] / sess["size"]
    faixa = (sess["size"] >= 10) & (sess["size"] <= 30)
    return {
        "visualizacoes": n,
        "usuarios": int(b["user_id"].nunique()),
        "frac_repeticoes": float(rep.mean()),
        "repeticoes_por_tipo": {k: float(v) for k, v in tipo.value_counts(normalize=True).items()},
        "frac_revisitas": float(rev.mean()),
        "frac_usuarios_com_revisita": float(b.loc[rev, "user_id"].nunique() / b["user_id"].nunique()),
        "top10_frac_visualizacoes": float(b["video_id"].isin(top).mean()),
        "top10_frac_revisitas": float(b.loc[rev, "video_id"].isin(top).mean()),
        "gini_visualizacoes_por_item": gini(vis_item),
        "gini_revisitas_por_item": gini(rev_item),
        "razao_wr_revisita_mediana": float(razao.median()),
        "frac_revisitas_wr_maior": float((razao > 1).mean()),
        "sessoes_tam_ge_2": int(len(sess)),
        "repeticao_na_sessao_media": float(rep_sess.mean()),
        "frac_sessoes_com_repeticao": float((rep_sess > 0).mean()),
        "repeticao_na_sessao_media_tam_10_30": float(rep_sess[faixa].mean()),
        "sessoes_tam_10_30": int(faixa.sum()),
    }


def loop(data):
    s = pd.read_csv(os.path.join(data, "small_matrix.csv"), usecols=["watch_ratio"])["watch_ratio"].dropna()
    r = s.clip(upper=5)
    return {"celulas": int(len(s)), "frac_wr_maior_1": float((s > 1).mean()), "frac_wr_maior_5": float((s > 5).mean()),
            "frac_recompensa_acima_de_1": float((r - r.clip(upper=1)).sum() / r.sum())}


def politicas(traj, top):
    sem = []
    for f in sorted(glob.glob(os.path.join(traj, "*_traj_s*.csv"))):
        m = re.fullmatch(r"(.+)_traj_s(\d+)\.csv", os.path.basename(f))
        if not m or nome(m.group(1)) is None or not os.path.isfile(f + ".ok"):
            continue
        df = pd.read_csv(f, usecols=["epoch", "protocolo", "traj", "t", "item_raw"])
        df = df[df["protocolo"] == "FB"].sort_values(["epoch", "traj", "t"])
        df["repetido"] = df.duplicated(["epoch", "traj", "item_raw"])
        df["top10"] = df["item_raw"].isin(top)
        t = df.groupby(["epoch", "traj"]).agg(n=("item_raw", "size"), d=("item_raw", "nunique"))
        e = df.groupby("epoch").agg(passos=("item_raw", "size"), distintos=("item_raw", "nunique"))
        r = df[df["repetido"]]
        sem.append({"metodo": nome(m.group(1)), "semente": int(m.group(2)),
                    "repeticao": float((1 - t["d"] / t["n"]).mean()),
                    "top10_frac_passos": float(df["top10"].mean()),
                    "top10_frac_repetidos": float(r["top10"].mean()) if len(r) else np.nan,
                    "cobertura_com_repeticao": float((e["distintos"] / e["passos"]).mean()),
                    "cobertura_sem_repeticao": float((e["distintos"] / t.groupby("epoch")["d"].sum()).mean())})
        print(f"{sem[-1]['metodo']:14s} s{sem[-1]['semente']}: {len(t)} trajetórias FB")
    return pd.DataFrame(sem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--traj", default=None)
    p.add_argument("--out", default="results/revisitas")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    b = pd.read_csv(os.path.join(a.data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    n0 = len(b)
    b = b.drop_duplicates(["user_id", "video_id", "timestamp", "watch_ratio"])
    pop = b.groupby("video_id")["user_id"].nunique().sort_values(ascending=False)
    top = set(pop.index[: int(len(pop) * 0.1)])
    aval = set(pd.read_csv(os.path.join(a.data, "small_matrix.csv"), usecols=["user_id"])["user_id"])
    res = {"duplicatas_removidas": n0 - len(b), "frac_duplicatas": (n0 - len(b)) / n0,
           "A_usuarios_avaliacao": revisitas(b[b["user_id"].isin(aval)], top),
           "A_todos": revisitas(b, top), "B_loop_small_matrix": loop(a.data)}
    json.dump(res, open(os.path.join(a.out, "revisitas_dados.json"), "w"), indent=1, ensure_ascii=False)

    linhas = [f"Duplicatas exatas removidas do big_matrix: {res['duplicatas_removidas']:,} ({100 * res['frac_duplicatas']:.1f}%)",
              "", "A. Revisitas no big_matrix (sem duplicatas)", f"{'medida':45s} {'aval.':>10s} {'todos':>10s}"]
    ua, ut = res["A_usuarios_avaliacao"], res["A_todos"]
    for k in ["visualizacoes", "usuarios", "frac_repeticoes", "frac_revisitas", "frac_usuarios_com_revisita",
              "top10_frac_visualizacoes", "top10_frac_revisitas", "gini_visualizacoes_por_item",
              "gini_revisitas_por_item", "razao_wr_revisita_mediana", "frac_revisitas_wr_maior", "sessoes_tam_ge_2",
              "repeticao_na_sessao_media", "frac_sessoes_com_repeticao", "repeticao_na_sessao_media_tam_10_30",
              "sessoes_tam_10_30"]:
        f = (lambda v: f"{v:>10,}") if isinstance(ua[k], int) else (lambda v: f"{v:>10.3f}")
        linhas.append(f"{k:45s} {f(ua[k])} {f(ut[k])}")
    for k in ["imediata", "mesma_sessao", "outra_sessao"]:
        linhas.append(f"{'repetições: ' + k:45s} {ua['repeticoes_por_tipo'].get(k, 0):>10.3f} "
                      f"{ut['repeticoes_por_tipo'].get(k, 0):>10.3f}")
    lp = res["B_loop_small_matrix"]
    linhas += ["", "B. Loop dentro da visualização (small_matrix, células observadas)",
               f"  watch_ratio > 1: {lp['frac_wr_maior_1']:.3f}; > 5: {lp['frac_wr_maior_5']:.3f}; "
               f"parte da recompensa de avaliação acima de 1: {lp['frac_recompensa_acima_de_1']:.3f}"]

    if a.traj:
        sem = politicas(a.traj, top)
        sem.to_csv(os.path.join(a.out, "revisitas_por_semente.csv"), index=False)
        cols = ["repeticao", "top10_frac_passos", "top10_frac_repetidos", "cobertura_com_repeticao",
                "cobertura_sem_repeticao"]
        g = sem.groupby("metodo")[cols]
        resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
        resumo.to_csv(os.path.join(a.out, "revisitas_resumo.csv"))
        linhas += ["", "C. Protocolo FB: repetição das políticas (média ± dp entre sementes)",
                   f"Referência real (usuários de avaliação): repetição na sessão {ua['repeticao_na_sessao_media']:.3f} "
                   f"(sessões de 10 a 30: {ua['repeticao_na_sessao_media_tam_10_30']:.3f}); top 10% nas revisitas "
                   f"{ua['top10_frac_revisitas']:.3f}, nas visualizações {ua['top10_frac_visualizacoes']:.3f}",
                   f"{'Método':14s} {'n':>2s}" + "".join(f" {c:>24s}" for c in cols)]
        n = sem.groupby("metodo").size()
        for m in ordem(resumo.index):
            linhas.append(f"{m:14s} {n[m]:>2d}" + "".join(
                f" {resumo.loc[m, ('media', c)]:>14.3f} ± {resumo.loc[m, ('dp', c)]:.3f}" for c in cols))
    txt = "\n".join(linhas)
    open(os.path.join(a.out, "revisitas_saida.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
