"""traj_log.patch: o CSV reproduz exatamente as trajetórias que o Callback_Coverage_Count percorre (MCD do log),
com o VectorReplayBuffer do fork do tianshou; sem TRAJ_LOG, nada muda; arquivo existente é recusado.

Uso: python tests/test_traj.py <evaluator.py original> <evaluator.py com o patch> <pasta do tianshou> <saida.csv>
"""
import importlib.util, os, sys
import numpy as np, pandas as pd

ORIG, PATCHED, TIANSHOU, OUT = sys.argv[1:5]
sys.path.insert(0, TIANSHOU)
from tianshou.data import Batch, VectorReplayBuffer  # noqa: E402


def load(path, name, env):
    for k in ("TRAJ_LOG",):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class LBE:  # LabelEncoder mínimo: índice -> id original
    classes_ = np.arange(50) * 7 + 3
    def inverse_transform(self, x): return self.classes_[np.asarray(x)]


def fill_buffer(rng, n_env=6, max_len=12):
    """Uma trajetória por ambiente, como na avaliação do DORL (os ambientes não são reiniciados após done)."""
    buf = VectorReplayBuffer(1000, n_env)
    lens = rng.integers(1, max_len, n_env)
    users = rng.integers(0, 40, n_env)
    truth = {}
    starts = np.full(n_env, -1)
    for t in range(max_len):
        ids = np.flatnonzero(lens > t)
        if not len(ids): break
        act = rng.integers(0, 50, len(ids))
        rew = rng.random(len(ids)) * 5
        done = lens[ids] == t + 1
        obs = np.stack([users[ids], np.full(len(ids), 50)], 1)
        b = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs, info={}, policy={})
        ptr, _, _, ep_idx = buf.add(b, buffer_ids=ids)
        for k, i in enumerate(ids):
            if t == 0: starts[i] = ptr[k]
            truth.setdefault(i, []).append((users[i], act[k], rew[k]))
    order = rng.permutation(n_env)                 # results["idxs"] não segue a ordem dos ambientes
    return buf, starts[order], [truth[i] for i in order]


def walk_original(buffer, indices):
    """Cópia literal da caminhada de get_actions_feat (evaluator.py, Callback_Coverage_Count)."""
    num_tests = len(indices)
    live_mat = np.zeros([0, num_tests], dtype=bool)
    act_mat = np.zeros([0, num_tests], dtype=bool)
    is_end = np.zeros([num_tests], dtype=bool)
    while not all(is_end):
        acts = buffer.act[indices]
        done = buffer.done[indices]
        act_mat = np.vstack([act_mat, acts])
        live_mat = np.vstack([live_mat, ~is_end])
        is_end[done] = True
        indices = buffer.next(indices)
    return act_mat[live_mat]


rng = np.random.default_rng(0)
orig = load(ORIG, "orig", {})
assert not hasattr(orig, "_traj_write")
p0 = load(PATCHED, "p0", {})
assert p0._TRAJ_LOG is None                       # sem a variável o gancho não roda
if os.path.exists(OUT): os.remove(OUT)
p1 = load(PATCHED, "p1", {"TRAJ_LOG": OUT})
truths = {}
for epoch in (1, 2):
    for proto in ("FB", "NX_0", "NX_10"):
        buf, idxs, truth = fill_buffer(rng)
        p1._traj_write(epoch, proto, buf, idxs, LBE())
        truths[(epoch, proto)] = (buf, idxs, truth)

df = pd.read_csv(OUT)
for (epoch, proto), (buf, idxs, truth) in truths.items():
    d = df[(df.epoch == epoch) & (df.protocolo == proto)]
    # mesma multiconjunto de ações que a MCD do log usa (walk_original achata por passo, não por trajetória)
    assert sorted(d["item"]) == sorted(walk_original(buf, idxs.copy()).astype(int).tolist())
    for j, steps in enumerate(truth):
        dj = d[d.traj == j].sort_values("t")
        assert list(dj.t) == list(range(len(steps)))
        assert list(dj.user) == [s[0] for s in steps]
        assert list(dj.item) == [s[1] for s in steps]
        assert np.allclose(dj.reward, [s[2] for s in steps])
        assert list(dj.item_raw) == list(LBE.classes_[dj.item])

try:
    load(PATCHED, "p2", {"TRAJ_LOG": OUT})._traj_write(1, "FB", *fill_buffer(rng)[:2], None)
    raise AssertionError("arquivo existente deveria ser recusado")
except RuntimeError:
    pass
print(f"OK: {len(df)} passos gravados = trajetórias do buffer (usuário, item, recompensa, ordem); "
      f"mesmas ações da MCD; sem TRAJ_LOG o gancho não roda; arquivo existente é recusado")
