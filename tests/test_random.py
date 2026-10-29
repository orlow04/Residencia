import sys, runpy, io, contextlib, itertools, numpy as np, pandas as pd, pickle
from collections import Counter
SCR, DATA, DOM = sys.argv[1:4]
sys.argv = [SCR, "--data", DATA, "--domination", DOM, "--epochs", "2", "--out", "/dev/null"]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(SCR)
mat, lfs = g["mat"], g["list_feat_small"]

class Env:  # cópia literal de KuaiEnv.step / _determine_whether_to_leave / _add_action_to_history
    def __init__(self, N, thr, max_turn): self.num_leave_compute, self.leave_threshold, self.max_turn = N, thr, max_turn; self.mat = mat; self.list_feat_small = lfs
    def reset(self, user): self.cum_reward = 0; self.total_turn = 0; self.cur_user = user; self.action = None; self.history_action = {}; self.sequence_action = []
    def step(self, action):
        self.action = action
        t = self.total_turn
        done = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            done = True
        self._add_action_to_history(t, action)
        reward = self.mat[self.cur_user, action]
        self.cum_reward += reward
        self.total_turn += 1
        return None, reward, done, {}
    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))
        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True
        return False
    def _add_action_to_history(self, t, action):
        self.sequence_action.append(action); self.history_action[t] = action

def ref_traj(rng, protocol, N=4):  # laço de evaluator.interactive_evaluation
    env = Env(N, 0, 30); env.reset(int(rng.integers(0, mat.shape[0])))
    remove = protocol != "FB"; force_length = 10 if protocol == "NX_10" else 0
    acts, rew, done = [], 0.0, False
    while not done:
        a = int(rng.integers(0, mat.shape[1]))
        while remove and a in acts: a = int(rng.integers(0, mat.shape[1]))
        acts.append(a); _, r, done, _ = env.step(a); rew += r
        if done:
            if force_length > 0: done = False
            else: break
        if force_length > 0 and len(acts) >= force_length: break
    return acts, rew

for protocol in ("FB", "NX_0", "NX_10"):
    for N in (4,):
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(3000):
            s1, w1 = g["trajectory"](r1, protocol); s2, w2 = ref_traj(r2, protocol)
            assert s1 == s2 and np.isclose(w1, w2), (protocol, s1, s2)
# MCD: cópia de get_feat_dominate_dict com df_item_val como em load_category (+1; ausente = 0)
list_feat = pd.read_csv(f"{DATA}/item_categories.csv").feat.map(eval).to_list()
df_feat = pd.DataFrame(list_feat, columns=['feat0', 'feat1', 'feat2', 'feat3']); df_feat[df_feat.isna()] = -1; df_feat = (df_feat + 1).astype(int)
df_item_val = df_feat.loc[g["items"]]
sorted_items = pickle.load(open(DOM, "rb"))["feat"]
values = np.array([p[1] for p in sorted_items]); values = values / sum(values); cumsum = values.cumsum(); ind = 0
for v in cumsum:
    if v > 0.8: break
    ind += 1
if ind == 0: ind += 1
dominated_values = np.array([p[0] for p in sorted_items])[:ind]
rng = np.random.default_rng(3); acts_enc = rng.integers(0, mat.shape[1], 500)
feat_numpy = df_item_val.loc[g["items"][acts_enc]].filter(regex="^feat", axis=1).to_numpy().astype(int)
dom = np.zeros(len(feat_numpy), dtype=bool)
for value in dominated_values: dom = dom | (feat_numpy == value).sum(axis=1).astype(bool)
assert np.isclose(dom.sum() / len(feat_numpy), g["is_dom"][acts_enc].mean())
print("OK: trajetórias (FB, NX_0, NX_10), recompensas e MCD idênticas às cópias do código do DORL")
