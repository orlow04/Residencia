"""OPE: aplica as políticas treinadas (checkpoints do roda_ckpt.sh) ao log real (ope_dados.py) e estima o seu valor.

Para cada passo do log reconstrói o estado como o StateTrackerAvg2 do DORL no teste (state_tracker2.py:138;
window_size 3, reward_handle "cat", sem embedding de usuário) e calcula π_e(·|s), a saída softmax do actor, que é a
distribuição de onde a política sorteia a ação no teste (a2c2.py:150–185; sem deterministic_eval e sem ruído).
  t = 0:  s = [e_pad, 1]                                  (ramo reset: recompensa inicial 1, sem normalizar)
  t ≥ 1:  s = média das entradas [e(item), r_norm] dos min(t, 3) passos anteriores e, se t < 3, da entrada de
          preenchimento [e_pad, r_norm(1)]; r_norm(r) = min((r − r_min) / (r_max − r_min), 1), com r_min e r_max da
          matriz de avaliação (0 e 5 no KuaiRec), como no teste.
Rede: Net (MLP com ReLU) → Actor.last (linear) → softmax, lidos do state_dict (nenhuma classe do DORL é importada).
A política é avaliada sem remover itens já vistos (como no FB); o log real quase não repete itens na sessão.

Estimativas por checkpoint, com w = π_e(a|s) / π_β(a|s) e π_β por contagens (ope_dados.py):
  SNIPS (principal, validado em ope_contexto.py), IPS e DR (diagnóstico), DM, ESS, maior peso;
  IC de 95% do SNIPS por bootstrap de usuários; SNIPS só nos passos com t ≥ 3 (estado sem preenchimento).
Sem pesos (descrevem a política nos contextos reais): entropia média de π_e(·|s) em nº efetivo de itens, exposição
agregada (cobertura a 95% da massa, Gini) e fração da exposição nos 10% de itens mais frequentes no log.
Referências: política de registro (média da recompensa registrada) e uniforme (mesmos estimadores).

Uso (de dorl-workspace/DORL-codes, na 4090 ou onde estiverem os checkpoints):
  python ../../ope_politicas.py --log ope_log.npz --data environments/KuaiRec/data \
      --ckpt "saved_models/KuaiEnv-v0/A2C_with_emb/A2C_with_emb_CK_*-e*.pt" --out ../../results/ope
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ope import dm, dr, ess, ips, matriz_avaliacao, snips  # noqa: E402

NOMES = {"MBPO": "MBPO", "MOPO": "MOPO", "DORL": "DORL", "DORLconst": "DORL-const"}


def carrega(path):
    """(embedding dos itens com a linha de preenchimento, camadas do MLP, camada final) a partir do checkpoint."""
    ck = torch.load(path, map_location="cpu")
    st, pol = ck["state_tracker"], ck["policy"]
    emb = next(v for k, v in st.items() if k.endswith("feat_item.weight")).float()

    def camadas(padrao):
        idx = sorted({int(m.group(1)) for k in pol if (m := re.fullmatch(padrao + r"\.(\d+)\.weight", k))})
        pre = padrao.replace("\\", "")
        return [(pol[f"{pre}.{i}.weight"].float(), pol[f"{pre}.{i}.bias"].float()) for i in idx]

    mlp = camadas(r"actor\.preprocess\.model\.model")
    ultima = camadas(r"actor\.last\.model")
    if not mlp or len(ultima) != 1:
        raise SystemExit(f"estrutura inesperada em {path}; chaves do actor: {[k for k in pol if k.startswith('actor')]}")
    return emb, mlp, ultima[0]


def estado(emb, h, hr, t, r_min, r_max):
    """Estado de cada passo. h, hr: (n, 3) itens anteriores (−1 = não há) e recompensas; t: posição na sessão."""
    pad = emb.shape[0] - 1
    n, J = h.shape
    norm = lambda r: torch.clamp((r - r_min) / (r_max - r_min), max=1.0)   # noqa: E731
    real = h >= 0                                                          # (n, J)
    e = emb[torch.where(real, h, torch.full_like(h, pad))] * real.unsqueeze(-1)
    r = norm(hr) * real
    falta = (t < J).float().unsqueeze(-1)                                  # entra a entrada de preenchimento
    soma_e = e.sum(1) + falta * emb[pad]
    soma_r = r.sum(1, keepdim=True) + falta * norm(torch.tensor(1.0))
    k = real.sum(1, keepdim=True).float() + falta
    s = torch.cat([soma_e, soma_r], 1) / k
    ini = t == 0
    s[ini] = torch.cat([emb[pad], torch.ones(1)]).to(s)
    return s


def pi_e(s, mlp, ultima):
    x = s
    for w, b in mlp:
        x = torch.relu(x @ w.T + b)
    return torch.softmax(x @ ultima[0].T + ultima[1], dim=-1)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def estimativas(pe_a, vq, L, q, rng, n_boot):
    w = pe_a / L["pb"]
    r, t = L["r"], L["t"]
    out = {"SNIPS": snips(w, r), "IPS": ips(w, r), "DR": dr(w, r, q[L["a"]], vq), "DM": dm(vq),
           "ESS": ess(w), "peso_max": float(w.max()), "massa_no_log": float(pe_a.mean()),
           "SNIPS_marginal": snips(pe_a / L["pb_marg"], r)}
    m = t >= 3
    out["SNIPS_t_ge_3"] = snips(w[m], r[m])
    # bootstrap de usuários: reamostra as somas por usuário de w·r e de w
    u = L["u_idx"]
    swr = np.bincount(u, weights=w * r)
    sw = np.bincount(u, weights=w)
    idx = rng.integers(len(sw), size=(n_boot, len(sw)))
    b = swr[idx].sum(1) / sw[idx].sum(1)
    out["SNIPS_ic95"] = [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True, help="glob dos checkpoints (entre aspas)")
    p.add_argument("--lote", type=int, default=16384)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/ope")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    z = np.load(a.log)
    L = {k: z[k] for k in ("t", "a", "r", "pb", "pb_marg", "h", "hr")}
    L["r"] = L["r"].astype(float)
    L["u_idx"] = np.unique(z["user"], return_inverse=True)[1]
    n, n_i = len(L["a"]), len(z["items"])
    mat, _, _ = matriz_avaliacao(a.data)
    r_min, r_max = float(mat.min()), float(mat.max())
    cnt = np.bincount(L["a"], minlength=n_i).astype(float)
    q = np.where(cnt > 0, np.bincount(L["a"], weights=L["r"], minlength=n_i) / np.maximum(cnt, 1), L["r"].mean())
    top = np.argsort(-cnt)[: n_i // 10]

    ref = {"registro": {"valor": float(L["r"].mean()), "top10_frac": float(cnt[top].sum() / cnt.sum()),
                         "gini_exposicao": gini(cnt)},
           "uniforme": estimativas(np.full(n, 1 / n_i), np.full(n, q.mean()), L, q, rng, a.boot)}
    linhas = [f"Log: {n:,} passos; r_min = {r_min:g}, r_max = {r_max:g}; π_β por contagens",
              f"Política de registro: valor {ref['registro']['valor']:.3f}; top 10% {ref['registro']['top10_frac']:.3f}; "
              f"Gini da exposição {ref['registro']['gini_exposicao']:.3f}",
              f"Uniforme: SNIPS {ref['uniforme']['SNIPS']:.3f} (IC95 {ref['uniforme']['SNIPS_ic95'][0]:.3f}–"
              f"{ref['uniforme']['SNIPS_ic95'][1]:.3f}); ESS {ref['uniforme']['ESS']:,.0f}", ""]

    h = torch.from_numpy(L["h"].astype(np.int64))
    hr = torch.from_numpy(L["hr"].astype(np.float32))
    tt = torch.from_numpy(L["t"].astype(np.int64))
    at = torch.from_numpy(L["a"].astype(np.int64))
    qt = torch.from_numpy(q.astype(np.float32)).to(a.device)
    sem = []
    for f in sorted(glob.glob(a.ckpt)):
        m = re.search(r"CK_(MBPO|MOPO|DORLconst|DORL)_s(\d+)-e(\d+)\.pt$", os.path.basename(f))
        if not m:
            print(f"ignorado (nome não reconhecido): {f}")
            continue
        emb, mlp, ultima = carrega(f)
        emb = emb.to(a.device)
        mlp = [(w.to(a.device), b.to(a.device)) for w, b in mlp]
        ultima = (ultima[0].to(a.device), ultima[1].to(a.device))
        if ultima[0].shape[0] != n_i or emb.shape[0] != n_i + 1:
            raise SystemExit(f"{f}: {ultima[0].shape[0]} ações e {emb.shape[0]} embeddings; esperado {n_i} e {n_i + 1}")
        pe_a = np.empty(n)
        vq = np.empty(n)
        ent = 0.0
        expo = torch.zeros(n_i, dtype=torch.float64, device=a.device)
        with torch.no_grad():
            for i in range(0, n, a.lote):
                j = slice(i, i + a.lote)
                s = estado(emb, h[j].to(a.device), hr[j].to(a.device), tt[j].to(a.device), r_min, r_max)
                pr = pi_e(s, mlp, ultima)
                pe_a[j] = pr.gather(1, at[j].to(a.device).unsqueeze(1)).squeeze(1).cpu().numpy()
                vq[j] = (pr @ qt).cpu().numpy()
                ent += float(-(pr * torch.log(pr.clamp_min(1e-30))).sum())
                expo += pr.sum(0).double()
        expo = expo.cpu().numpy() / n
        d = {"metodo": NOMES[m.group(1)], "semente": int(m.group(2)), "epoca": int(m.group(3))}
        d |= estimativas(pe_a, vq, L, q, rng, a.boot)
        ordem = np.sort(expo)[::-1]
        d |= {"itens_efetivos_por_passo": float(np.exp(ent / n)), "top10_frac": float(expo[top].sum()),
              "gini_exposicao": gini(expo), "itens_para_95_da_exposicao": int(np.searchsorted(np.cumsum(ordem), 0.95) + 1)}
        sem.append(d)
        print(f"{d['metodo']:11s} s{d['semente']} (época {d['epoca']}): SNIPS {d['SNIPS']:.3f} "
              f"[{d['SNIPS_ic95'][0]:.3f}, {d['SNIPS_ic95'][1]:.3f}], ESS {d['ESS']:,.0f}, peso máx. {d['peso_max']:.1f}")
    if not sem:
        raise SystemExit(f"nenhum checkpoint reconhecido em {a.ckpt}")
    df = pd.DataFrame(sem)
    df.to_csv(os.path.join(a.out, "ope_por_semente.csv"), index=False)
    cols = ["SNIPS", "SNIPS_t_ge_3", "SNIPS_marginal", "IPS", "DR", "DM", "ESS", "peso_max", "massa_no_log",
            "itens_efetivos_por_passo", "itens_para_95_da_exposicao", "top10_frac", "gini_exposicao"]
    g = df.groupby("metodo")[cols]
    resumo = pd.concat({"media": g.mean(), "dp": g.std(ddof=1)}, axis=1)
    resumo.to_csv(os.path.join(a.out, "ope_resumo.csv"))
    json.dump({"config": vars(a), "referencias": ref}, open(os.path.join(a.out, "ope_referencias.json"), "w"),
              indent=1, ensure_ascii=False)
    linhas.append(f"{'Método':11s} {'n':>2s}" + "".join(f" {c:>26s}" for c in cols))
    for met in [x for x in NOMES.values() if x in resumo.index]:
        linhas.append(f"{met:11s} {int((df['metodo'] == met).sum()):>2d}" + "".join(
            f" {resumo.loc[met, ('media', c)]:>14.3f} ± {resumo.loc[met, ('dp', c)]:<9.3f}" for c in cols))
    txt = "\n".join(linhas)
    open(os.path.join(a.out, "ope_saida.txt"), "w").write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
