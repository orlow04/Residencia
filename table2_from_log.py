"""Resume um log de treino do DORL-codes no formato da Tabela 2 de Gao et al. (2023).

Reproduz a agregação do script oficial (results_for_paper/visual_utils.py, handle_one_col
com final_rate=1): média ± desvio padrão sobre as 200 épocas de treino, no protocolo
"No Overlapping" (prefixo NX_0_ no log). Esse ± é a dispersão entre épocas de UMA execução,
não entre sementes.

Validado contra os logs originais dos autores (results_for_paper/results_all/kuairec):
reproduz exatamente os valores da Tabela 2 para MBPO, MOPO, DORL, UCB e epsilon-greedy.

Mapeamento log -> Tabela 2:
    NX_0_R_tra      -> R_tra   (recompensa acumulada na trajetória)
    NX_0_ctr        -> R_each  (recompensa média por rodada)
    NX_0_len_tra    -> Length  (comprimento da trajetória)
    NX_0_ifeat_feat -> MCD     (majority category domination)

Uso:
    python table2_from_log.py saved_models/KuaiEnv-v0/A2C_with_emb/logs/[MBPO]_*.log \
        --dataset KuaiRec --method MBPO
    python table2_from_log.py <log> --all-protocols   # também FB e NX_10 (útil para o experimento de protocolos)
"""
import argparse
import ast
import re

import pandas as pd

# Gao et al. (2023), Tabela 2: (média, desvio)
TABLE2 = {
    "KuaiRec": {
        "UCB": {"R_tra": (3.606, 0.609), "R_each": (0.853, 0.114), "Length": (4.219, 0.389), "MCD": (0.811, 0.058)},
        "epsilon-greedy": {"R_tra": (3.515, 0.731), "R_each": (0.828, 0.129), "Length": (4.219, 0.405), "MCD": (0.823, 0.048)},
        "MBPO": {"R_tra": (12.043, 1.312), "R_each": (0.770, 0.029), "Length": (15.646, 1.637), "MCD": (0.362, 0.047)},
        "MOPO": {"R_tra": (11.427, 1.750), "R_each": (0.892, 0.051), "Length": (12.809, 1.850), "MCD": (0.479, 0.062)},
        "DORL": {"R_tra": (20.494, 2.671), "R_each": (0.767, 0.026), "Length": (26.712, 3.419), "MCD": (0.379, 0.015)},
    },
    "KuaiRand": {
        "UCB": {"R_tra": (1.651, 0.152), "R_each": (0.372, 0.028), "Length": (4.431, 0.212), "MCD": (0.789, 0.024)},
        "epsilon-greedy": {"R_tra": (1.711, 0.126), "R_each": (0.351, 0.025), "Length": (4.880, 0.270), "MCD": (0.773, 0.024)},
        "MBPO": {"R_tra": (10.933, 0.946), "R_each": (0.431, 0.021), "Length": (25.345, 1.819), "MCD": (0.306, 0.040)},
        "MOPO": {"R_tra": (10.934, 0.963), "R_each": (0.437, 0.019), "Length": (25.002, 1.891), "MCD": (0.343, 0.029)},
        "DORL": {"R_tra": (11.850, 1.036), "R_each": (0.428, 0.022), "Length": (27.609, 2.121), "MCD": (0.296, 0.036)},
    },
}
NAMES = {"R_tra": "R_tra", "ctr": "R_each", "len_tra": "Length", "ifeat_feat": "MCD"}
PROTOCOLS = {"NX_0_": "No Overlapping (Tabela 2)", "": "Free (FB)", "NX_10_": "No Overlapping, 10 rodadas forçadas"}

LINE = re.compile(r"Epoch: \[(-?\d+)\], Info: \[(\{.*\})\]")
ARRAY = re.compile(r"array\(\[([^\]]*)\]\)")


def read_log(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            info = ast.literal_eval(ARRAY.sub(r"\1", m.group(2)))
            rows[int(m.group(1))] = {k: float(v) for k, v in info.items() if _is_number(v)}
    if not rows:
        raise SystemExit(f"Nenhuma linha 'Epoch: [..], Info: [..]' encontrada em {path}")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def summarize(df, prefix):
    out = {}
    for key, name in NAMES.items():
        col = prefix + key
        if col in df:
            out[name] = (df[col].mean(), df[col].std())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--dataset", choices=TABLE2)
    p.add_argument("--method", choices=["UCB", "epsilon-greedy", "MBPO", "MOPO", "DORL"])
    p.add_argument("--all-protocols", action="store_true")
    a = p.parse_args()

    df = read_log(a.log)
    # Os baselines (UCB, epsilon-greedy) registram uma avaliação antes do treino, como época -1.
    # A Tabela 2 usa só as 200 épocas de treino; descartá-la reproduz os números do artigo exatamente.
    df = df[df.index >= 0]
    print(f"Log: {a.log}\nÉpocas usadas: {len(df)} (de {df.index.min()} a {df.index.max()})\n")

    protocols = PROTOCOLS if a.all_protocols else {"NX_0_": PROTOCOLS["NX_0_"]}
    for prefix, label in protocols.items():
        res = summarize(df, prefix)
        print(f"== {label}")
        ref = TABLE2.get(a.dataset, {}).get(a.method) if prefix == "NX_0_" else None
        for name, (m, s) in res.items():
            line = f"  {name:7s} {m:9.3f} ± {s:.3f}"
            if ref:
                rm, rs = ref[name]
                z = (m - rm) / rs if rs else float("nan")
                line += f"   | artigo {rm:.3f} ± {rs:.3f}   diferença = {z:+.2f} desvios do artigo"
            print(line)
        print()
    if a.method and a.dataset:
        print("Atenção: o ± do artigo é a dispersão entre épocas de uma execução. "
              "Para afirmar reprodutibilidade, repita com outras sementes e compare as médias.")


if __name__ == "__main__":
    main()
