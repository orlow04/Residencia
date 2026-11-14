"""OPE: log real para avaliar as políticas (big_matrix → passos (contexto, ação, recompensa) com π_β estimada).

Os usuários de avaliação não têm interação com os itens do small_matrix no big_matrix, então o log usa os demais
usuários. Passos:
1. big_matrix sem duplicatas exatas, só usuários fora da avaliação; sessões pelo intervalo > 30 min na sequência
   completa do usuário (todos os itens).
2. Dentro de cada sessão ficam só os itens do small_matrix (os únicos que a política conhece), em ordem de timestamp;
   t é a posição nessa sequência filtrada.
3. Para cada passo guarda-se o que o estado da política usa (StateTrackerAvg2, state_tracker2.py:138, window_size 3,
   reward_handle "cat"): os até 3 itens anteriores da sessão e suas recompensas. Ação = item visto (índice na ordem do
   LabelEncoder do small_matrix, a mesma do ambiente); recompensa = watch_ratio limitado a 5 (escala da avaliação).
4. Política de registro estimada pelas contagens do próprio log, como validado em ope_contexto.py:
     t ≥ 1: π_β(a | l) = (c[l, a] + β·marg(a)) / (c[l]·1 + β), l = item anterior da sessão
     t = 0: π_β(a) = (c0[a] + β·marg(a)) / (c0·1 + β), c0 = contagens dos primeiros itens de sessão
   com marg(a) = frequência suavizada do item no log e β = 10. Guarda também a versão marginal (diagnóstico).

Saída: <out>/ope_log.npz (não versionado; ~100 MB) e <resumo>/ope_dados.{txt,json}.
Uso (de dorl-workspace/DORL-codes):
  python ../../ope_dados.py --data environments/KuaiRec/data --out . --resumo ../../results/ope
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ope import matriz_avaliacao  # noqa: E402

SESSAO, JANELA = 1800, 3


def passos(data, users, items):
    b = pd.read_csv(os.path.join(data, "big_matrix.csv"), usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    n0 = len(b)
    b = b.drop_duplicates()
    b = b[~b["user_id"].isin(set(users))].sort_values(["user_id", "timestamp"], kind="mergesort")
    nova = (b["user_id"] != b["user_id"].shift()) | (b["timestamp"].diff() > SESSAO)
    b["sessao"] = nova.cumsum()
    n_fora = len(b)
    b = b[b["video_id"].isin(set(items))].reset_index(drop=True)
    b["a"] = np.searchsorted(items, b["video_id"].to_numpy())
    b["r"] = b["watch_ratio"].clip(upper=5).fillna(b["watch_ratio"].clip(upper=5).mean())
    b["t"] = b.groupby("sessao").cumcount()
    info = {"linhas_big": n0, "linhas_fora_da_avaliacao": n_fora, "passos": len(b),
            "usuarios": int(b["user_id"].nunique()), "sessoes": int(b["sessao"].nunique())}
    return b, info


def historico(b):
    """h[i, k] = índice do (k+1)-ésimo item anterior na sessão (−1 se não há); hr = recompensas correspondentes."""
    n = len(b)
    a, r, t = b["a"].to_numpy(), b["r"].to_numpy(), b["t"].to_numpy()
    h = np.full((n, JANELA), -1, dtype=np.int32)
    hr = np.zeros((n, JANELA), dtype=np.float32)
    for k in range(1, JANELA + 1):
        ok = np.flatnonzero(t >= k)          # linhas da mesma sessão são contíguas: o k-ésimo anterior é a linha i − k
        h[ok, k - 1] = a[ok - k]
        hr[ok, k - 1] = r[ok - k]
    return h, hr


def propensoes(a, t, l, n_i, beta):
    marg = np.bincount(a, minlength=n_i).astype(float)
    marg_n = (marg + 1) / (marg.sum() + n_i)
    pb = np.empty(len(a))
    ini = t == 0
    c0 = np.bincount(a[ini], minlength=n_i).astype(float)
    pb[ini] = (c0[a[ini]] + beta * marg_n[a[ini]]) / (c0.sum() + beta)
    seq = ~ini
    chave = l[seq].astype(np.int64) * n_i + a[seq]
    c = np.bincount(chave, minlength=n_i * n_i)
    c_l = np.bincount(l[seq], minlength=n_i).astype(float)
    pb[seq] = (c[chave] + beta * marg_n[a[seq]]) / (c_l[l[seq]] + beta)
    return pb, marg_n[a], c[chave]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--out", default=".")
    p.add_argument("--resumo", default="results/ope")
    a = p.parse_args()
    os.makedirs(a.resumo, exist_ok=True)
    mat, users, items = matriz_avaliacao(a.data)
    n_i = len(items)
    b, info = passos(a.data, users, items)
    h, hr = historico(b)
    act, t, r = b["a"].to_numpy(), b["t"].to_numpy(), b["r"].to_numpy()
    pb, pb_marg, c_par = propensoes(act, t, h[:, 0], n_i, a.beta)
    np.savez_compressed(os.path.join(a.out, "ope_log.npz"), sessao=b["sessao"].to_numpy(), user=b["user_id"].to_numpy(),
                        t=t.astype(np.int32), a=act.astype(np.int32), r=r.astype(np.float32), h=h, hr=hr,
                        pb=pb, pb_marg=pb_marg, items=items)

    tam = b.groupby("sessao").size()
    seq = t >= 1
    info |= {"beta": a.beta, "itens": n_i, "itens_no_log": int(len(np.unique(act))),
             "recompensa_media_do_registro": float(r.mean()),
             "recompensa_media_small_matrix": float(mat.mean()),
             "frac_t0": float((t == 0).mean()), "frac_t_menor_3": float((t < JANELA).mean()),
             "sessao_tam_mediana": float(tam.median()), "sessao_tam_media": float(tam.mean()),
             "sessao_tam_p90": float(tam.quantile(0.9)), "frac_passos_em_sessoes_ate_30": float(tam[tam <= 30].sum() / len(b)),
             "frac_pares_com_contagem_1": float((c_par == 1).mean()),
             "pb_quantis": {q: float(np.quantile(pb, float(q))) for q in ("0.01", "0.1", "0.5", "0.9", "0.99")},
             "pb_marg_quantis": {q: float(np.quantile(pb_marg, float(q))) for q in ("0.01", "0.1", "0.5", "0.9", "0.99")},
             "peso_uniforme_ess": float((1 / pb).sum() ** 2 / ((1 / pb) ** 2).sum()),
             "passos_t_ge_1": int(seq.sum())}
    json.dump(info, open(os.path.join(a.resumo, "ope_dados.json"), "w"), indent=1, ensure_ascii=False)
    linhas = [
        f"big_matrix: {info['linhas_big']:,} linhas; fora da avaliação (sem duplicatas): {info['linhas_fora_da_avaliacao']:,}",
        f"Passos do log (itens do small_matrix): {info['passos']:,} em {info['sessoes']:,} sessões de {info['usuarios']:,} usuários; "
        f"{info['itens_no_log']} dos {n_i} itens aparecem",
        f"Sessão (itens filtrados): mediana {info['sessao_tam_mediana']:.0f}, média {info['sessao_tam_media']:.1f}, "
        f"p90 {info['sessao_tam_p90']:.0f}; passos em sessões de até 30 itens: {100 * info['frac_passos_em_sessoes_ate_30']:.1f}%",
        f"Passos em t = 0: {100 * info['frac_t0']:.1f}%; em t < 3 (estado com o item de preenchimento): {100 * info['frac_t_menor_3']:.1f}%",
        f"Recompensa média do registro (valor da política de registro): {info['recompensa_media_do_registro']:.3f}; "
        f"média do small_matrix: {info['recompensa_media_small_matrix']:.3f}",
        f"π_β por contagens (β = {a.beta:g}), quantis 1/10/50/90/99%: " + " / ".join(f"{v:.2e}" for v in info["pb_quantis"].values()),
        f"π_β marginal, quantis 1/10/50/90/99%: " + " / ".join(f"{v:.2e}" for v in info["pb_marg_quantis"].values()),
        f"Pares (anterior → item) com contagem 1: {100 * info['frac_pares_com_contagem_1']:.1f}% dos passos com t ≥ 1",
        f"ESS de uma política-alvo uniforme (pesos ∝ 1/π_β): {info['peso_uniforme_ess']:,.0f}",
    ]
    txt = "\n".join(linhas)
    open(os.path.join(a.resumo, "ope_dados.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
