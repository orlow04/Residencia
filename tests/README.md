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
| `test_ope_politicas.py` | `ope_politicas.py` | as classes reais do DORL (`StateTrackerAvg2`, `Actor` e buffer do tianshou; exige o `dorl-workspace`): estado e π_e(·|s) de uma sessão de 7 passos |
| `test_mecanismo.py` | `mecanismo_bolha.py`, `fb_rotacao.py` | regra contrafactual × cópia literal de `_determine_whether_to_leave`; V condicional × reimplementação; medidas do FB em casos montados |
| `test_sessoes.py` | `sessoes_reais.py` | elegibilidade (cadeia contígua), repetição de categoria e fim de sessão calculados linha a linha |
| `test_e6.py` | `e6_saida_prob.patch` aplicado ao `KuaiEnv.py` | o `_determine_whether_to_leave` original: idêntico e sem consumir números aleatórios sem `E6_P_REP`; com as variáveis, frequência de saída = P_REP/P_BASE conforme repita ou não; reprodutível |
| `test_tmax.py` | `tmax.patch` aplicado ao `run_Policy_Main.py` | `MAX_R − MIN_R` de `simulated_env.py:81–82` com o `entropy_min/max` de `prepare_envs`: intacto sem `TMAX_LAMBDA`; com a variável, igual ao `train_max` da execução com o outro `lambda_entropy`, nos dois sentidos; o patch só acrescenta linhas |
| `test_e5.py` | `e5_alvo.patch` aplicado ao `KuaiEnv.py` | o `KuaiEnv.get_df_kuairec` original, **com os dados reais** (o carregador lê categorias e características): idêntico sem `E5_RAW_WR`; com a variável, só o alvo muda |

## Como rodar

Da raiz do repositório, com o `dorl-workspace` criado pelo `setup.sh`:

```bash
bash tests/roda_testes.sh
```

O script gera os dados sintéticos numa pasta temporária e lê os arquivos originais do DORL-codes do git, de modo que
o estado dos patches no `dorl-workspace` não importa. Para cada patch, compara o original com uma cópia com o patch
aplicado; os três patches do `KuaiEnv.py` (E5, A5 e E6) também são testados juntos. O `test_e5` exige os dados
reais em `dorl-workspace/DORL-codes/environments/KuaiRec/data` e é pulado sem eles. Cada teste imprime uma linha
começando com `OK:` ou falha com `AssertionError`; o script termina com `TODOS OS TESTES PASSARAM` ou com a lista
dos que falharam. Os comandos de cada teste, para rodar um a um, estão no próprio `roda_testes.sh`.

Última execução: 17/11/2026, no Mac, 16 de 16 testes passaram (~35 s).
