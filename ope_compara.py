"""OPE × simulador: lê results/ope/ope_por_semente.csv (ope_politicas.py) e results/e1/e1_por_semente.csv
(e1_protocolos.py) e resume (a) cada método sob OPE contra a política de registro, (b) quanto a estimativa depende dos
pesos (SNIPS × DM) e (c) a concordância, política a política, entre o SNIPS e as métricas do simulador.

Uso (da raiz do repositório):
  python ope_compara.py --ope results/ope --e1 results/e1/e1_por_semente.csv
Gera results/ope/ope_comparacao.txt. Precisa de numpy, pandas e scipy.
"""
import argparse
import ast
import json
import os

import pandas as pd
from scipy.stats import spearmanr

ORDEM = ["MBPO", "MOPO", "DORL", "DORL-const"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ope", default="results/ope")
    p.add_argument("--e1", default="results/e1/e1_por_semente.csv")
    a = p.parse_args()
    o = pd.read_csv(os.path.join(a.ope, "ope_por_semente.csv"))
    ref = json.load(open(os.path.join(a.ope, "ope_referencias.json")))["referencias"]
    reg, uni = ref["registro"]["valor"], ref["uniforme"]
    ic = o.SNIPS_ic95.map(ast.literal_eval)
    o["lo"], o["hi"] = ic.str[0], ic.str[1]
    L = [f"Políticas: {len(o)}; épocas dos checkpoints: {sorted(o.epoca.unique().tolist())}",
         f"Referências: política de registro {reg:.3f}; uniforme SNIPS {uni['SNIPS']:.3f} "
         f"(IC95 {uni['SNIPS_ic95'][0]:.3f}–{uni['SNIPS_ic95'][1]:.3f}), ESS {uni['ESS']:,.0f}", "",
         f"{'Método':11s} {'SNIPS média ± dp':>17s} {'mediana':>8s} {'mín–máx':>13s} {'ESS mín–máx':>13s} "
         f"{'itens ef./passo':>16s} {'itens p/ 95%':>13s}  top 10% por semente"]
    for m in [x for x in ORDEM if x in set(o.metodo)]:
        g = o[o.metodo == m]
        L.append(f"{m:11s} {g.SNIPS.mean():>8.3f} ± {g.SNIPS.std(ddof=1):<6.3f} {g.SNIPS.median():>8.3f} "
                 f"{g.SNIPS.min():>6.3f}–{g.SNIPS.max():<6.3f} {g.ESS.min():>6.0f}–{g.ESS.max():<6.0f} "
                 f"{g.itens_efetivos_por_passo.min():>7.2f}–{g.itens_efetivos_por_passo.max():<8.2f} "
                 f"{g.itens_para_95_da_exposicao.min():>5d}–{g.itens_para_95_da_exposicao.max():<7d}  "
                 + " ".join(f"{v:.3f}" for v in sorted(g.top10_frac)))
    med = o.groupby("metodo").SNIPS.mean()
    L += ["", f"Maior diferença entre médias de métodos: {med.max() - med.min():.3f}; "
              f"menor dp entre sementes: {o.groupby('metodo').SNIPS.std(ddof=1).min():.3f}",
          f"Contra a política de registro ({reg:.3f}): SNIPS abaixo em {(o.SNIPS < reg).sum()} de {len(o)}; "
          f"IC95 inteiro abaixo em {(o.hi < reg).sum()}, inteiro acima em {(o.lo > reg).sum()}, "
          f"contém em {((o.lo <= reg) & (o.hi >= reg)).sum()}",
          "Políticas cujo IC95 não fica inteiro abaixo: " + "; ".join(
              f"{r.metodo} s{r.semente} {r.SNIPS:.3f} [{r.lo:.3f}, {r.hi:.3f}] ESS {r.ESS:,.0f}"
              for r in o[o.hi >= reg].itertuples()),
          f"SNIPS × DM (sem pesos): Spearman {spearmanr(o.SNIPS, o.DM)[0]:.3f}; "
          f"|SNIPS − DM| mediana {(o.SNIPS - o.DM).abs().median():.3f}, máximo {(o.SNIPS - o.DM).abs().max():.3f}",
          f"Massa de π_e na ação registrada: {o.massa_no_log.min():.6f}–{o.massa_no_log.max():.6f} "
          f"(uniforme: {uni['massa_no_log']:.6f})", "",
          "Concordância com o simulador, política a política (Spearman entre o SNIPS e a métrica do simulador da mesma",
          "semente; o roda_ckpt.sh confere os CK_* contra as execuções originais; p sem correção para 8 comparações):"]
    e = pd.read_csv(a.e1)
    e["semente"] = pd.to_numeric(e.log.str.extract(r"_s(\d+)\]")[0], errors="coerce")
    e = e[e.metodo.isin(set(o.metodo)) & e.semente.notna()]
    for prot in ["NX_0", "NX_10", "FB"]:
        for met in ["R_each", "R_tra", "Length"]:
            if prot == "NX_10" and met == "Length":
                continue                                                   # fixo em 10
            s = e[(e.protocolo == prot) & (e.metrica == met)].groupby(["metodo", "semente"]).valor.mean()
            j = o.merge(s.rename("sim").reset_index(), on=["metodo", "semente"])
            if len(j) != len(o):
                raise SystemExit(f"{prot} {met}: {len(j)} políticas casadas de {len(o)}")
            rho, pv = spearmanr(j.SNIPS, j.sim)
            L.append(f"  {prot:5s} {met:6s} ρ = {rho:+.2f} (p = {pv:.2f}, n = {len(j)}); médias no simulador: "
                     + " < ".join(j.groupby("metodo").sim.mean().sort_values().index))
    L.append("  médias do SNIPS: " + " < ".join(med.sort_values().index))
    txt = "\n".join(L)
    open(os.path.join(a.ope, "ope_comparacao.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
