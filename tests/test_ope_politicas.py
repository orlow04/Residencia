"""Teste do ope_politicas.py contra as classes reais do DORL (não cópias): estado e π_e(·|s) de uma sessão.

Monta um StateTrackerAvg2 (window_size 3, reward_handle "cat", sem embedding de usuário), um Actor do tianshou
(Net 64-64 + softmax) com pesos aleatórios e um buffer do tianshou com uma sessão de 7 passos. Para cada passo t,
compara:
  - o estado do DORL (t = 0: ramo reset; t ≥ 1: forward com o buffer, como o get_emb na coleta de teste) com o
    estado() do ope_politicas.py a partir de (itens anteriores, recompensas, t);
  - a saída do Actor com o pi_e() do ope_politicas.py, lendo os pesos de um checkpoint gravado no formato do
    save_model_fn (util/utils.py:147).
Inclui recompensa acima de r_max (corte em 1) e r_min ≠ 0.
Uso: python tests/test_ope_politicas.py "$PWD/ope_politicas.py" "$PWD/dorl-workspace/DORL-codes"
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn

script, dorl = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(dorl, p) for p in ("src", "src/DeepCTR-Torch", "src/tianshou")]
from core.inputs import get_dataset_columns  # noqa: E402
from core.state_tracker2 import StateTrackerAvg2  # noqa: E402
from tianshou.data import Batch, VectorReplayBuffer  # noqa: E402
from tianshou.utils.net.common import Net  # noqa: E402
from tianshou.utils.net.discrete import Actor  # noqa: E402

spec = importlib.util.spec_from_file_location("ope_politicas", script)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)

torch.manual_seed(0)
rng = np.random.default_rng(0)
n_u, n_i, d, J = 5, 40, 8, 3
r_min, r_max = 0.5, 5.0
saved = nn.ModuleDict({"feat_user": nn.Embedding(n_u, d), "feat_item": nn.Embedding(n_i, d)})
ucols, acols, fcols, *_ = get_dataset_columns(d, d, n_u, n_i, envname="KuaiEnv-v0")
tracker = StateTrackerAvg2(ucols, acols, fcols, d, saved, train_max=9.0, train_min=0.0, test_max=r_max, test_min=r_min,
                           reward_handle="cat", device="cpu", use_userEmbedding=False, window_size=J)
actor = Actor(Net(d + 1, hidden_sizes=[64, 64], device="cpu"), n_i, device="cpu")

with tempfile.TemporaryDirectory() as tmp:
    ck = os.path.join(tmp, "A2C_with_emb_CK_DORL_s0-e200.pt")
    torch.save({"policy": {f"actor.{k}": v for k, v in actor.state_dict().items()},
                "state_tracker": tracker.state_dict()}, ck)
    emb, mlp, ultima = op.carrega(ck)

user = 3
itens = rng.choice(n_i, 7, replace=False)
recs = np.array([0.2, 6.0, 1.3, 0.0, 4.9, 2.2, 0.7])              # 6.0 > r_max: normalizada e cortada em 1
buf = VectorReplayBuffer(64, 1)
pior_s = pior_p = 0.0
for t in range(len(itens)):
    if t == 0:
        s_dorl = tracker.forward(obs=np.array([[user, n_i]]), reset=True)
    else:
        ind = buf.last_index[~buf[buf.last_index].done]
        s_dorl = tracker.forward(buffer=buf, indices=ind, reset=False, is_obs=False, is_train=False)
    h = np.full((1, J), -1, dtype=np.int64)
    hr = np.zeros((1, J), dtype=np.float32)
    for k in range(1, min(t, J) + 1):
        h[0, k - 1], hr[0, k - 1] = itens[t - k], recs[t - k]
    s_meu = op.estado(emb, torch.from_numpy(h), torch.from_numpy(hr), torch.tensor([t]), r_min, r_max)
    pior_s = max(pior_s, float((s_dorl.detach().float() - s_meu).abs().max()))
    p_dorl, _ = actor(s_dorl.detach().float())
    pior_p = max(pior_p, float((p_dorl.detach() - op.pi_e(s_meu, mlp, ultima)).abs().max()))
    obs = np.array([[user, n_i if t == 0 else itens[t - 1]]])
    buf.add(Batch(obs=obs, act=np.array([itens[t]]), rew=np.array([recs[t]]), terminated=np.array([False]),
                  truncated=np.array([False]), done=np.array([False]), obs_next=np.array([[user, itens[t]]]),
                  info=np.array([{}])), buffer_ids=[0])

assert pior_s < 1e-5, pior_s
assert pior_p < 1e-6, pior_p
print(f"OK: estado e π_e(·|s) idênticos aos do StateTrackerAvg2 e do Actor do DORL em 7 passos "
      f"(maior diferença: estado {pior_s:.1e}, probabilidade {pior_p:.1e})")
