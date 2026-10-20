"""Valida os dados do KuaiRec processados pela pipeline contra a Tabela 1 de Gao et al. (2023).

Rode DEPOIS de `python preprocessing_kuairec.py`. Também calcula o SHA-256 dos arquivos
brutos, para registrar a versão exata dos dados usada nos experimentos.

    python validate_kuairec.py --data DORL-codes/environments/KuaiRec/data
"""
import argparse
import ast
import hashlib
import json
import os

import pandas as pd

# Tabela 1 do DORL (linha KuaiRec)
EXPECTED = {
    "train": {"users": 7176, "items": 10728, "interactions": 12530806, "categories": 31},
    "test": {"users": 1411, "items": 3327, "interactions": 4676570, "categories": 31},
}
RAW = ["big_matrix.csv", "small_matrix.csv", "user_features.csv", "item_categories.csv"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stats(df, item_cats):
    items = df["item_id"].unique()
    cats = set()
    for i in items:
        cats.update(item_cats.get(i, []))
    return {"users": df["user_id"].nunique(), "items": len(items),
            "interactions": len(df), "categories": len(cats)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="pasta environments/KuaiRec/data")
    p.add_argument("--out", default="kuairec_validation.json")
    a = p.parse_args()

    cats = pd.read_csv(os.path.join(a.data, "item_categories.csv"))
    item_cats = dict(zip(cats["video_id"], cats["feat"].map(ast.literal_eval)))

    cols = ["user_id", "item_id", "watch_ratio_normed", "duration_normed"]
    got = {
        "train": stats(pd.read_csv(os.path.join(a.data, "big_matrix_processed.csv"), usecols=cols), item_cats),
        "test": stats(pd.read_csv(os.path.join(a.data, "small_matrix_processed.csv"), usecols=cols), item_cats),
    }

    ok_all = True
    print(f"{'conjunto':8s} {'campo':13s} {'obtido':>12s} {'Tabela 1':>12s}")
    for split, exp in EXPECTED.items():
        for k, v in exp.items():
            ok = got[split][k] == v
            ok_all &= ok
            print(f"{split:8s} {k:13s} {got[split][k]:12,d} {v:12,d}  {'OK' if ok else 'DIVERGE'}")

    t = got["test"]
    print(f"\nDensidade da small_matrix: {t['interactions'] / (t['users'] * t['items']):.4f}")

    hashes = {f: sha256(os.path.join(a.data, f)) for f in RAW if os.path.exists(os.path.join(a.data, f))}
    with open(a.out, "w") as f:
        json.dump({"stats": got, "expected": EXPECTED, "match": ok_all, "sha256": hashes}, f, indent=2)
    print(f"\nResultado: {'TODOS OS CAMPOS CONFEREM' if ok_all else 'HÁ DIVERGÊNCIAS: investigue antes de treinar'}")
    print(f"Estatísticas e hashes gravados em {a.out}")


if __name__ == "__main__":
    main()
