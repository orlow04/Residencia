# Testes de fidelidade dos scripts de análise

Cada teste compara um script deste repositório com uma **cópia literal** do trecho correspondente do
DORL-codes (commit `818c9bd`), em dados sintéticos. Nenhum teste usa o KuaiRec real nem GPU.
Requisitos: numpy, pandas, scipy e scikit-learn (o venv do DORL já tem).

| Teste | Script testado | Compara com |
|---|---|---|
| `test_ref.py` | `diagnostico_pe.py` | `user_model_ensemble.get_save_entropy_mat`, laço de `simulated_env._compute_pred_reward`, `entropy_min/max` de `run_Policy_Main.py` |
| `test_patch.py` | `pe_log.patch` aplicado ao `simulated_env.py` | o `simulated_env.py` original: recompensa idêntica sem variáveis e com `PE_LOG`; `PE_CONST` altera |
| `test_evmat.py` | `r_hat_vs_avaliacao.py` | `KuaiEnv.load_mat` (duplicatas, células ausentes, NaN, valores > 5) |
| `test_random.py` | `politica_aleatoria.py` | `KuaiEnv.step` / `_determine_whether_to_leave`, laço de `evaluator.interactive_evaluation`, `get_feat_dominate_dict` |

## Como rodar (da raiz deste repositório)

```bash
python make_synthetic_kuairec.py --out /tmp/synth --n_users 500 --n_items 120 \
    --small_items 100 --small_users 50 --per_user 80
python tests/prepara_dados_teste.py /tmp/synth /tmp/rtest

python tests/test_ref.py    "$PWD/diagnostico_pe.py"     /tmp/synth
python tests/test_evmat.py  "$PWD/r_hat_vs_avaliacao.py" /tmp/rtest
python tests/test_random.py "$PWD/politica_aleatoria.py" /tmp/rtest /tmp/rtest/dom.pickle

# patch: compara o arquivo original (do git) com o arquivo já alterado
git -C dorl-workspace/DORL-codes show HEAD:src/core/worldModel/simulated_env.py > /tmp/simenv_orig.py
python tests/test_patch.py /tmp/simenv_orig.py \
    dorl-workspace/DORL-codes/src/core/worldModel/simulated_env.py /tmp/synth /tmp/pe_log_teste.csv
```

Cada teste imprime uma linha começando com `OK:` ou falha com `AssertionError`.
