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
| `test_traj.py` | `traj_log.patch` aplicado ao `evaluator.py` | caminhada de `Callback_Coverage_Count.get_actions_feat` sobre o `VectorReplayBuffer` do fork do tianshou |
| `test_bolha.py` | `bolha.py` | a definição de V e D calculada item a item (não há código do DORL a copiar) |
| `test_a5.py` | `a5_janela.patch` aplicado ao `KuaiEnv.py` | o `_determine_whether_to_leave` original: idêntico sem `A5_FIX`; com a variável, janela = min(t, N) anteriores; o recorte original ocorre em N/2 < t < N |
| `test_alem.py` | `alem_da_acuracia.py` | cada critério (R_each, MCD, V, cobertura, Gini, novidade, serendipidade, personalização) calculado item a item |
| `test_ope.py` | `ope.py` | propriedades exatas dos estimadores num problema pequeno: IPS e DR sem viés (esperança exata), DR com q̂ exato sem variância |
| `test_mecanismo.py` | `mecanismo_bolha.py`, `fb_rotacao.py` | regra contrafactual × cópia literal de `_determine_whether_to_leave`; V condicional × reimplementação; medidas do FB em casos montados |
| `test_sessoes.py` | `sessoes_reais.py` | elegibilidade (cadeia contígua), repetição de categoria e fim de sessão calculados linha a linha |
| `test_e6.py` | `e6_saida_prob.patch` aplicado ao `KuaiEnv.py` | o `_determine_whether_to_leave` original: idêntico e sem consumir números aleatórios sem `E6_P_REP`; com as variáveis, frequência de saída = P_REP/P_BASE conforme repita ou não; reprodutível |
| `test_e5.py` | `e5_alvo.patch` aplicado ao `KuaiEnv.py` | o `KuaiEnv.get_df_kuairec` original, **com os dados reais** (o carregador lê categorias e características): idêntico sem `E5_RAW_WR`; com a variável, só o alvo muda |

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

# trajetórias: original (do git) × cópia com o traj_log.patch
git -C dorl-workspace/DORL-codes show HEAD:src/core/evaluation/evaluator.py > /tmp/eval_orig.py
cp /tmp/eval_orig.py /tmp/eval_patch.py && patch -s /tmp/eval_patch.py traj_log.patch
python tests/test_traj.py /tmp/eval_orig.py /tmp/eval_patch.py dorl-workspace/DORL-codes/src/tianshou /tmp/traj_teste.csv

# bolha: V, D e janelas do usuário contra a definição item a item
python tests/test_bolha.py "$PWD/bolha.py"

# critérios além da acurácia; mecanismo da diversidade e FB
python tests/test_alem.py "$PWD/alem_da_acuracia.py"
python tests/test_mecanismo.py "$PWD/mecanismo_bolha.py" "$PWD/fb_rotacao.py"

# avaliação off-policy (branch off-policy)
python tests/test_ope.py "$PWD/ope.py"

python tests/test_sessoes.py "$PWD/sessoes_reais.py"

# A5: regra de saída original (do git) × cópia com o a5_janela.patch (de dorl-workspace/DORL-codes)
(cd dorl-workspace/DORL-codes && git show HEAD:environments/KuaiRec/env/KuaiEnv.py > /tmp/kuai_orig.py \
  && cp /tmp/kuai_orig.py /tmp/kuai_a5.py && patch -s /tmp/kuai_a5.py ../../a5_janela.patch \
  && python ../../tests/test_a5.py /tmp/kuai_orig.py /tmp/kuai_a5.py)

# E6: mesma estrutura do A5, com o e6_saida_prob.patch (testar também com os três patches aplicados juntos)
(cd dorl-workspace/DORL-codes && cp /tmp/kuai_orig.py /tmp/kuai_e6.py && patch -s /tmp/kuai_e6.py ../../e6_saida_prob.patch \
  && python ../../tests/test_e6.py /tmp/kuai_orig.py /tmp/kuai_e6.py)

# E5: carregador original (do git) × cópia com o e5_alvo.patch, nos dados reais (de dorl-workspace/DORL-codes)
# Rode duas vezes se video_duration_normed.csv ainda não existir: a primeira chamada cria esse cache e a
# duração lida do CSV difere da calculada em memória no último dígito.
cd dorl-workspace/DORL-codes
git show HEAD:environments/KuaiRec/env/KuaiEnv.py > /tmp/kuai_orig.py
cp /tmp/kuai_orig.py /tmp/kuai_patch.py && patch -s /tmp/kuai_patch.py ../../e5_alvo.patch
python ../../tests/test_e5.py /tmp/kuai_orig.py /tmp/kuai_patch.py environments/KuaiRec/data
```

Cada teste imprime uma linha começando com `OK:` ou falha com `AssertionError`.
