import importlib.util, os, sys, types, csv
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

ORIG, PATCHED, DATA, OUT = sys.argv[1:5]
# stubs para as dependências pesadas
gym = types.ModuleType("gym"); gym.Env = object; sys.modules["gym"] = gym
torch = types.ModuleType("torch"); torch.FloatTensor = object; sys.modules["torch"] = torch
tq = types.ModuleType("tqdm"); tq.tqdm = lambda x, **k: x; sys.modules["tqdm"] = tq
core = types.ModuleType("core"); util = types.ModuleType("core.util")
util.clip0 = lambda x: np.amax(x, 0); util.compute_action_distance = util.compute_exposure = None
sys.modules["core"] = core; sys.modules["core.util"] = util

def load(path, name, env):
    for k in ("PE_LOG", "PE_CONST"): os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

# tabela de entropia a partir dos dados sintéticos (mesma lógica do DORL)
df = pd.read_csv(f"{DATA}/big_matrix.csv").rename(columns={"video_id": "item_id"}).sort_values(["user_id", "timestamp"])
cnt = defaultdict(lambda: defaultdict(int)); last, hist = -1, []
for u, i in df[["user_id", "item_id"]].to_numpy():
    if u != last: last, hist = u, []
    for L in (1, 2):
        if len(hist) >= L: cnt[tuple(sorted(hist[-L:]))][i] += 1
    hist.append(i)
def ent(d):
    if len(d) <= 1: return 1
    p = np.array(list(d.values())) / sum(d.values()); return -np.sum(p * np.log2(p)) / np.log2(len(d))
emap = {k: ent(v) for k, v in cnt.items()}
emap = {k: v for j, (k, v) in enumerate(emap.items()) if j % 7}   # remove chaves para forçar ausências
small = pd.read_csv(f"{DATA}/small_matrix.csv")
lbe = LabelEncoder().fit(small["video_id"].unique())

class Task: lbe_item = lbe
def make_self(rng, T=30):
    s = types.SimpleNamespace(env_name="KuaiEnv-v0", entropy_window=[1, 2], step_n_actions=2, env_task=Task(),
        entropy_dict={"map": emap}, lambda_variance=0.05, lambda_entropy=5.0, MIN_R=7.0, version="v1",
        predicted_mat=rng.normal(0.5, 0.3, (50, len(lbe.classes_))), maxvar_mat=np.abs(rng.normal(0.2, 0.1, (50, len(lbe.classes_)))),
        cur_user=np.array([rng.integers(50)]), history_action=np.zeros(T, dtype=int), total_turn=0)
    return s

def run(mod, seed=0, n=40, T=30):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = make_self(rng, T); acts = rng.integers(0, len(lbe.classes_), T)
        for t in range(T):
            s.total_turn = t; s.history_action[t] = acts[t]
            out.append(mod.SimulatedEnv._compute_pred_reward(s, 0, acts[t]))
    return np.array(out)

r_orig = run(load(ORIG, "orig", {}))
r_same = run(load(PATCHED, "p0", {}))
assert np.array_equal(r_orig, r_same), "sem variáveis, o patch mudou a recompensa"
if os.path.exists(OUT): os.remove(OUT)
mlog = load(PATCHED, "p1", {"PE_LOG": OUT}); r_log = run(mlog); mlog._pe_state["fh"].flush()
assert np.array_equal(r_orig, r_log), "com PE_LOG, o patch mudou a recompensa"
rows = list(csv.DictReader(open(OUT)))
assert len(rows) == len(r_orig) and len({r["env"] for r in rows}) == 40
assert all(r["n_completados"] == ("1" if r["turn"] == "0" else "0") for r in rows)
assert any(int(r["n_ausentes"]) > 0 for r in rows)
# r_penalizado do log reproduz a recompensa final (version v1, sem exposição): clip0(r_pen)
assert np.allclose([max(float(r["r_penalizado"]), 0) for r in rows], r_orig)
r_c = run(load(PATCHED, "p2", {"PE_CONST": "1.5"}))
assert not np.array_equal(r_orig, r_c)
print(f"OK: idêntico ao original sem variáveis e com PE_LOG; log com {len(rows)} linhas; PE_CONST altera a recompensa")
print(open(OUT).read().splitlines()[0]); print(open(OUT).read().splitlines()[1])
