"""Teste do ope.py: propriedades exatas dos estimadores num problema pequeno (não há código do DORL a copiar).

1. IPS sem viés: a esperança exata do IPS (soma sobre todos os pares x, a com probabilidade p(x)·π_β(a|x)) é igual ao
   valor verdadeiro Σ_x p(x) Σ_a π_e(a|x) r(x, a).
2. DR com q̂ = r exato: o estimador dá o valor verdadeiro em qualquer amostra (variância zero).
3. DR com π_β correta e q̂ qualquer: esperança exata igual ao valor verdadeiro.
4. SNIPS é invariante a multiplicar todos os pesos por uma constante; IPS com corte infinito = IPS.
5. ESS = n quando todos os pesos são iguais.
Uso: python tests/test_ope.py "$PWD/ope.py"
"""
import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("ope", sys.argv[1])
ope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ope)

rng = np.random.default_rng(0)
n_x, n_a = 7, 5
px = rng.dirichlet(np.ones(n_x))
pb = rng.dirichlet(np.ones(n_a), size=n_x)
pe = rng.dirichlet(np.ones(n_a) * 0.3, size=n_x)
r = rng.uniform(0, 5, size=(n_x, n_a))
q = r + rng.normal(0, 1, size=(n_x, n_a))                      # modelo direto errado
verdade = float(px @ (pe * r).sum(axis=1))

# todos os pares (x, a), com a probabilidade de cada um no log
X, A = np.meshgrid(np.arange(n_x), np.arange(n_a), indexing="ij")
X, A = X.ravel(), A.ravel()
prob = px[X] * pb[X, A]
w = ope.pesos(pe[X, A], pb[X, A])
e_ips = float(np.sum(prob * w * r[X, A]))
assert abs(e_ips - verdade) < 1e-12, (e_ips, verdade)

v_q = (pe * q).sum(axis=1)[X]
termos_dr = v_q + w * (r[X, A] - q[X, A])
e_dr = float(np.sum(prob * termos_dr))
assert abs(e_dr - verdade) < 1e-12, (e_dr, verdade)

# amostra: DR com q̂ = r exato não depende da amostra
idx = rng.choice(len(X), size=500, p=prob)
xs, as_ = X[idx], A[idx]
ws = ope.pesos(pe[xs, as_], pb[xs, as_])
v_r = (pe * r).sum(axis=1)
assert abs(ope.dr(ws, r[xs, as_], r[xs, as_], v_r[xs]) - ope.dm(v_r[xs])) < 1e-12
assert abs(ope.snips(3 * ws, r[xs, as_]) - ope.snips(ws, r[xs, as_])) < 1e-12
assert abs(ope.ips_corte(ws, r[xs, as_], np.inf) - ope.ips(ws, r[xs, as_])) < 1e-12
assert abs(ope.ess(np.full(500, 2.0)) - 500) < 1e-9

print("OK: IPS e DR sem viés (esperança exata); DR com q̂ exato sem variância; SNIPS invariante à escala; "
      "IPS com corte infinito = IPS; ESS = n com pesos iguais")
