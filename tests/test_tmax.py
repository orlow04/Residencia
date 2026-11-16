"""tmax.patch: sem TMAX_LAMBDA, o train_max não muda; com TMAX_LAMBDA = x, passa a ser o MAX_R − MIN_R de uma execução
com lambda_entropy = x (cópia literal de simulated_env.py:81–82 e do entropy_min/max de prepare_envs), nos dois
sentidos (MOPO → escala do DORL; DORL → escala do MOPO). Confere também que o patch só acrescenta linhas e que a
chamada fica atrás de `if _TMAX_LAMBDA is not None`.

Extrai _tmax_ajusta do arquivo com o patch pela árvore sintática (não importa o run_Policy_Main.py inteiro).
Uso: python tests/test_tmax.py <run_Policy_Main.py original> <run_Policy_Main.py com o patch>
"""
import ast
import difflib
import os
import sys

import numpy as np

ORIG, PATCHED = sys.argv[1:3]
fonte = open(PATCHED).read()
arv = ast.parse(fonte)
fn = next(n for n in arv.body if isinstance(n, ast.FunctionDef) and n.name == "_tmax_ajusta")


class Registro:
    def info(self, *a, **k):
        pass


def carrega(valor):
    os.environ.pop("TMAX_LAMBDA", None)
    if valor is not None:
        os.environ["TMAX_LAMBDA"] = str(valor)
    ns = {"os": os, "logger": Registro()}
    exec("_TMAX_LAMBDA = os.environ.get('TMAX_LAMBDA')", ns)
    exec(compile(ast.Module([fn], []), PATCHED, "exec"), ns)
    os.environ.pop("TMAX_LAMBDA", None)
    return ns["_tmax_ajusta"]


def train_max_original(predicted_mat, maxvar_mat, mapa, entropy_window, lambda_variance, lambda_entropy):
    # prepare_envs (run_Policy_Main.py:170–176)
    entropy_set = set(entropy_window)
    entropy_min = 0
    entropy_max = 0
    if len(entropy_set):
        for entropy_term in entropy_set:
            entropy_min += min([v for k, v in mapa.items() if len(k) == entropy_term])
            entropy_max += max([v for k, v in mapa.items() if len(k) == entropy_term])
    # SimulatedEnv.__init__ (simulated_env.py:81–82)
    MIN_R = predicted_mat.min() - lambda_variance * maxvar_mat.max() + lambda_entropy * entropy_min
    MAX_R = predicted_mat.max() - lambda_variance * maxvar_mat.min() + lambda_entropy * entropy_max
    # setup_state_tracker (run_Policy_Main.py:250)
    return MAX_R - MIN_R


rng = np.random.default_rng(0)
pior = 0.0
for caso in range(200):
    pm, vm = rng.normal(size=(20, 30)), rng.random((20, 30))
    janela = [[1, 2], [1], [2], [1, 2, 3]][caso % 4]
    mapa = {}
    for termo in janela:
        for _ in range(int(rng.integers(1, 40))):
            mapa[tuple(sorted(rng.choice(500, termo, replace=False)))] = float(rng.random())
    mapa.setdefault((0, 1, 2, 3), 0.5)                          # chave de outro tamanho: não pode entrar na faixa
    lv = float(rng.choice([0, 0.05, 1]))
    la, lb = (0.0, 5.0) if caso % 2 else (float(rng.random() * 10), float(rng.random() * 10))
    ta = train_max_original(pm, vm, mapa, janela, lv, la)
    tb = train_max_original(pm, vm, mapa, janela, lv, lb)
    sem = carrega(None)(ta, mapa, janela, la)
    assert sem is ta, "sem TMAX_LAMBDA o valor tem de voltar intacto"
    pior = max(pior, abs(carrega(lb)(ta, mapa, janela, la) - tb),     # execução com λ = la, escala de λ = lb
               abs(carrega(la)(tb, mapa, janela, lb) - ta),           # sentido inverso
               abs(carrega(la)(ta, mapa, janela, la) - ta))           # TMAX_LAMBDA = λ: nada muda
assert pior < 1e-9, pior

# O patch só acrescenta linhas, e a única chamada fica atrás do teste da variável.
dif = [l for l in difflib.ndiff(open(ORIG).read().splitlines(), fonte.splitlines()) if l[:2] in ("- ", "+ ")]
assert dif and all(l.startswith("+ ") for l in dif), "o patch removeu ou alterou linhas do original"
linhas = fonte.splitlines()
chamadas = [i for i, l in enumerate(linhas) if "_tmax_ajusta(" in l and not l.startswith("def ")]
assert len(chamadas) == 1 and linhas[chamadas[0] - 1].strip() == "if _TMAX_LAMBDA is not None:", chamadas
print(f"OK: 200 casos; sem TMAX_LAMBDA o train_max volta intacto; com a variável, igual ao MAX_R − MIN_R da execução "
      f"com o outro lambda_entropy (maior diferença {pior:.1e}); o patch só acrescenta {len(dif)} linhas")
