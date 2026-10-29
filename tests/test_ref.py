"""Compara diagnostico_pe.py com cópias literais do código do DORL (818c9bd)."""
import runpy, sys, io, contextlib
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

SCR = sys.argv[1]; DATA = sys.argv[2]
sys.argv = [SCR, "--data", DATA, "--n_traj", "10", "--out", "/dev/null"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(SCR)

# --- cópia de user_model_ensemble.get_save_entropy_mat (entropy_window = [1, 2]) ---
def get_entropy(mylist, need_count=True):
    if len(mylist) <= 1:
        return 1
    cnt_dict = Counter(mylist) if need_count else mylist
    prob = np.array(list(cnt_dict.values())) / sum(cnt_dict.values())
    log_prob = np.log2(prob)
    return - np.sum(log_prob * prob) / np.log2(len(cnt_dict))

df_train = pd.read_csv(f"{DATA}/big_matrix.csv").rename(columns={"video_id": "item_id"})
df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
map_hist_count = defaultdict(lambda: defaultdict(int)); lastuser = -1
def update_map(m, hist_tra, item, require_len):
    if len(hist_tra) < require_len: return
    m[tuple(sorted(hist_tra[-require_len:]))][item] += 1
hist_tra = []
for (user, item, time) in df_uit.to_numpy():
    user = int(user); item = int(item)
    if user != lastuser: lastuser = user; hist_tra = []
    for rl in {1, 2}: update_map(map_hist_count, hist_tra, item, rl)
    hist_tra.append(item)
map_entropy = {k: get_entropy(v, need_count=False) for k, v in map_hist_count.items()}

# 1) tabela idêntica
K1, E1, K2, E2, M = g["K1"], g["E1"], g["K2"], g["E2"], g["M"]
ref1 = {k[0]: v for k, v in map_entropy.items() if len(k) == 1}
ref2 = {k[0] * M + k[1]: v for k, v in map_entropy.items() if len(k) == 2}
assert len(ref1) == len(K1) and len(ref2) == len(K2), (len(ref1), len(K1), len(ref2), len(K2))
assert np.allclose([ref1[k] for k in K1], E1) and np.allclose([ref2[k] for k in K2], E2)
# 2) entropy_min / max como em run_Policy_Main.py
emin = sum(min(v for k, v in map_entropy.items() if len(k) == L) for L in {1, 2})
emax = sum(max(v for k, v in map_entropy.items() if len(k) == L) for L in {1, 2})
assert np.isclose(emin, g["entropy_min"]) and np.isclose(emax, g["entropy_max"])

# 3) P_E por passo: cópia do laço de simulated_env._compute_pred_reward
df_small = pd.read_csv(f"{DATA}/small_matrix.csv").rename(columns={"video_id": "item_id"})
lbe_item = LabelEncoder().fit(df_small["item_id"].unique())
assert np.array_equal(lbe_item.classes_, g["S"])
rng = np.random.default_rng(1); T = 30; step_n_actions = 2
for _ in range(300):
    acts = rng.integers(0, len(g["S"]), T)
    history_action = np.zeros(T, dtype=int)
    for total_turn in range(T):
        history_action[total_turn] = acts[total_turn]
        action_k = history_action[max(0, total_turn - step_n_actions + 1):total_turn + 1]
        action_reverse = lbe_item.inverse_transform(action_k)[::-1]
        entropy = 0
        for k in range(len(action_reverse)):
            s = tuple(sorted(action_reverse[:k + 1]))
            entropy += map_entropy[s] if s in map_entropy else 1
        if len(action_reverse) < step_n_actions:
            entropy += step_n_actions - len(action_reverse)
        # versão do script
        A = g["S"][acts]
        v1, _, _ = g["lookup"](K1, E1, g["D1"], A[total_turn:total_turn + 1])
        if total_turn == 0: v2 = 1.0
        else: v2, _, _ = g["lookup"](K2, E2, g["D2"], g["pair_key"](A[total_turn:total_turn + 1], A[total_turn - 1:total_turn]))
        assert np.isclose(entropy, float(v1[0] + np.asarray(v2).ravel()[0])), (total_turn, entropy, v1, v2)
print("OK: tabela (k=1: %d chaves, k=2: %d), entropy_min/max e P_E por passo idênticos ao código original" % (len(K1), len(K2)))
