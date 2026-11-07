"""e5_alvo.patch: sem E5_RAW_WR, KuaiEnv.get_df_kuairec devolve exatamente o mesmo que o original; com a variável,
só a coluna do alvo muda, e passa a ser o watch_ratio bruto limitado a 5 (como em KuaiEnv.load_mat).

Usa os dados reais do KuaiRec (o carregador lê também categorias e características de usuário e item).
Uso (de dorl-workspace/DORL-codes):
  python ../../tests/test_e5.py <KuaiEnv.py original> <KuaiEnv.py com o patch> environments/KuaiRec/data [arquivo]
"""
import importlib.util, os, sys
import numpy as np, pandas as pd

ORIG, PATCHED, DATA = sys.argv[1:4]
NAME = sys.argv[4] if len(sys.argv) > 4 else "small_matrix_processed.csv"
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.DATAPATH = os.path.abspath(DATA)
    return mod


os.environ.pop("E5_RAW_WR", None)
orig = load(ORIG, "kuai_orig").KuaiEnv.get_df_kuairec(NAME)
patched_mod = load(PATCHED, "kuai_patch")
sem = patched_mod.KuaiEnv.get_df_kuairec(NAME)
for a, b in zip(orig, sem):                      # df_data, df_user, df_item, list_feat
    assert (a.equals(b) if hasattr(a, "equals") else a == b), "sem a variável o resultado mudou"

os.environ["E5_RAW_WR"] = "1"
com = patched_mod.KuaiEnv.get_df_kuairec(NAME)
os.environ.pop("E5_RAW_WR")
d0, d1 = orig[0], com[0]
outras = [c for c in d0.columns if c != "watch_ratio_normed"]
assert d0[outras].equals(d1[outras]), "colunas além do alvo mudaram"
wr = pd.read_csv(os.path.join(DATA, NAME), usecols=["watch_ratio"])["watch_ratio"].clip(upper=5).to_numpy()
assert np.array_equal(d1["watch_ratio_normed"].to_numpy(), wr)
assert not np.array_equal(d0["watch_ratio_normed"].to_numpy(), wr)
print(f"OK: {NAME}, {len(d1):,} linhas; sem E5_RAW_WR idêntico ao original; com a variável, só o alvo muda "
      f"(watch_ratio limitado a 5: média {wr.mean():.3f}, máx {wr.max():.1f})")
