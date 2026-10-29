"""Resume o CSV gravado com PE_LOG (patch pe_log.patch) durante o treino do DORL.

Responde: quanto do P_E é preenchimento (chave ausente -> 1), quanto ele varia em comparação com r_hat,
e qual constante usar no E2 (média do P_E na fase final do treino).

Uso: python analisa_pe_log.py pe_log_DORL_s2.csv [--lambda_entropy 5] [--fases 5]
"""
import argparse

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--lambda_entropy", type=float, default=5.0)
p.add_argument("--fases", type=int, default=5, help="divide o treino em partes iguais, pela ordem das linhas")
a = p.parse_args()

cols = {"env": "int32", "turn": "int16", "PE": "float32", "n_ausentes": "int8", "n_completados": "int8",
        "r_hat": "float32"}
df = pd.read_csv(a.csv, usecols=list(cols), dtype=cols)   # só o necessário: ~25 bytes por linha
df["consultas"] = 2 - df["n_completados"]          # chaves procuradas na tabela (k=1 e k=2)
df["fase"] = np.minimum(np.arange(len(df)) * a.fases // len(df), a.fases - 1) + 1
print(f"{len(df):,} passos de treino; {df['env'].nunique()} ambientes\n")


def resumo(d):
    s = d[d["turn"] >= 1]
    return pd.Series({
        "passos": len(d),
        "PE_medio": d["PE"].mean(),
        "PE_medio_t>=1": s["PE"].mean(),
        "PE_dp_t>=1": s["PE"].std(),
        "frac_consultas_ausentes": d["n_ausentes"].sum() / d["consultas"].sum(),
        "frac_passos_t0": (d["turn"] == 0).mean(),
        "dp_lambda2*PE_t>=1": a.lambda_entropy * s["PE"].std(),
        "dp_r_hat_t>=1": s["r_hat"].std(),
        "corr(PE, r_hat)_t>=1": s["PE"].corr(s["r_hat"]),
    })


tab = df.groupby("fase").apply(resumo)
tab.loc["total"] = resumo(df)
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
    print(tab.T.to_string())

ultima = df[df["fase"] == a.fases]
print(f"\nConstante para o E2 (PE_CONST): média do P_E na fase {a.fases} de {a.fases} = {ultima['PE'].mean():.4f}"
      f"  (no treino inteiro: {df['PE'].mean():.4f})")
