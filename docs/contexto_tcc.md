# Contexto do TCC — histórico, decisões e detalhes técnicos

Atualizado em 29/10/2026 (semana 8, dia do gate). Complementa o `CLAUDE.md`.

## 1. Trajetória da residência (do Termo de Aceite)

| Semana | O que foi feito |
|---|---|
| 1 | Leitura dos tópicos da CSCE; escolha de RL e Sistemas de Recomendação; início de fundamentos e algoritmos de RL |
| 2 | Fim do estudo dirigido de RL; início de RS e avaliação de RS |
| 3 | Fim do estudo inicial das duas áreas; busca e leitura de revisões da interseção (RLRS) |
| 4 | Framework comum de RLRS segundo as revisões (Afsar et al.) |
| 5 | Trilhas de pesquisa a seguir |
| 6 | Escolha do artigo para replicação (DORL) |
| 7 | Reprodução dos baselines (MBPO, UCB, ε-greedy) e validação do fluxo |
| 8 | Pergunta orientadora; S1, S2, S3; E1 parcial; DORL e MOPO com 5 sementes; registro do P_E e E2 em andamento |

Fundamentação da especialização: (1) trajetória desde um projeto de recomendador com BERT como codificador de avaliações e histórico, passando por deep learning avançado e RL; (2) afinidade com as ligações do RL com psicologia comportamental e neurociência (Sutton & Barto).

## 2. Foco e decisões tomadas

- Interesse central: o lado técnico de como recomendação guiada por comportamento afeta pessoas — quais dados comportamentais os modelos usam e quais objetivos (por exemplo, exploração) moldam o que é recomendado.
- Trilhas dentro de RLRS: exploração (linha REINFORCE do Google/YouTube) e aplicação em plataformas com forte presença humana (avaliação online; como RL molda a relação dos usuários com algoritmos).
- Descartados: gamificação adaptativa (foco antigo); trilha de explicabilidade (outubro, com apoio do Cedric, por se afastar da especialização); "redes sociais" como enquadramento (prematuro).
- RL se justifica sobre bandits contextuais pela dinâmica temporal de estado (habituação, fadiga de novidade); essa distinção deve ser preservada.
- Avaliação off-policy (IPS, doubly robust) é pilar teórico para avaliar políticas sem usuário no laço.
- Por que o DORL: dados públicos e código aberto; contrasta bandits e RL no mesmo ambiente; mede concentração (MCD); virou baseline de ROLeR (CIKM 2024) e DARLR (SIGIR 2025), que mantêm o termo de entropia sem analisá-lo.
- Objetivo da reprodução: instanciar o framework de RLRS de Afsar et al. e ter uma base funcional para experimentos sobre a pergunta orientadora, mais que reproduzir todos os números.
- Decomposição da Justificativa: **objeto** (concentração da exposição produzida pela política), **mecanismo** (objetivo de treino e métrica de avaliação), **condição** (protocolos em que o efeito aparece, some ou se inverte).

## 3. Literatura âncora

- Sutton & Barto (fundamentos de RL).
- Revisões de RLRS: Afsar et al. (2022, ACM CSUR) e Lin et al. (IEEE TNNLS). Há 19 artigos recentes que citam uma ou ambas, organizados em grupos de leitura (planilha "Artigos Residência").
- Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, cap. 9.
- Aggarwal, *Recommender Systems: The Textbook*, capítulo de avaliação.
- Avaliação off-policy: IPS, doubly robust.
- Para "bolha" (a ler; confirmar referências): Michiels et al. (UMAP '22 Adjunct, revisão conceitual de filter bubbles); Michiels et al. (2023, "How Should We Measure Filter Bubbles?"); revisão sistemática arXiv 2307.01221; Chaney, Stewart & Engelhardt (RecSys 2018); Jiang et al. (AIES 2019); Nguyen et al. (WWW 2014).

## 4. O DORL em resumo (da Justificativa de Replicação)

- Problema: RL offline superestima o valor de pares (s, a) mal cobertos pelo log; o conservadorismo corrige, mas suprime itens raros e produz efeito Mateus.
- Método (Eq. 9): r̃ = r̂ − λ₁·P_U + λ₂·P_E. P_U = máximo da variância num ensemble de DeepFMs gaussianos; P_E motivado como −KL(π_β ‖ uniforme) (Eq. 8).
- Garantia (Teorema 4.1, do MOPO) é condicional: o DORL não prova que P_U + P_E satisfaz a Eq. 6.
- Afirmações e força da evidência:

| Afirmação | Evidência no artigo | Força |
|---|---|---|
| C1. Conservadorismo por incerteza produz efeito Mateus | Fig. 3; BCQ com MCD alto | Moderada; o CQL tem MCD menor que o MOPO no KuaiRec |
| C2. Efeito Mateus prejudica a experiência de longo prazo | Seção 3: retenção D+1 cai com a taxa de repetição | Correlacional |
| C3. A causa é suprimir amostras incertas | Lema 5.1 + Eq. 7 | Passo central afirmado, não derivado |
| C4. A entropia reintroduz exploração | Fig. 8 (varredura de λ₂) | Só em simulador; possível confusão com bônus de sobrevivência (testada pelo E2) |

Correspondência com os nomes antigos da Justificativa: E1 (inversão por protocolo) = E1; E2 (P_E constante) = E2; E3 (sensibilidade à regra de saída) = adiado; E4 (concentração por usuário) = item "bolha" da próxima semana.

## 5. Detalhes de implementação verificados no código (DORL-codes `818c9bd`)

- **Regra de saída** (`KuaiEnv._determine_whether_to_leave`): em t ≥ 1, conta as categorias dos itens em `sequence_action[t−N:t]`; sai se alguma categoria do item atual tiver contagem > `leave_threshold` (0). A checagem ocorre antes de o item atual entrar no histórico, e o passo que dispara a saída ainda conta em Length e na recompensa. Com t − N negativo, o fatiamento recorta a janela: com N = 4, em t = 3 a janela tem só 1 item. Máximo de 30 rodadas (`t ≥ max_turn − 1`).
- **Protocolos** (`collector_set.py`): FB (`remove_recommended_ids=False`, `force_length=0`); NX_0 (`True`, `0`); NX_10 (`True`, `10`; o `done` do ambiente é ignorado até 10 rodadas).
- **Métricas** (`util/utils.py`, `evaluator.py`), por época de 100 trajetórias: Length = passos/100; R_tra = recompensa total/100; R_each = R_tra/Length; MCD = fração dos itens recomendados (somando as trajetórias) com alguma categoria dominante.
- **Categorias dominantes:** `environments/KuaiRec/env/feature_domination.pickle` (distribuído no repositório; codificação +1 do `load_category`). Com `top_rate` 0,8 são 10 categorias: 29, 9, 12, 7, 10, 27, 8, 6, 18, 2 (codificação +1).
- **Tabela de entropia** (`user_model_ensemble.get_save_entropy_mat`): para cada usuário, em ordem de `timestamp`, chave = último item (k=1) ou par ordenado dos dois últimos (k=2); valor = entropia normalizada (log₂ dividido por log₂ do número de continuações distintas) da distribuição do próximo item; uma única continuação → 1.
- **P_E no passo** (`simulated_env._compute_pred_reward`): soma sobre k=1 e k=2 das chaves formadas pelo item atual e o anterior (IDs originais via `lbe_item.inverse_transform`); chave ausente → +1; em t = 0 falta o par → +1.
- **Deslocamento:** `MIN_R = predicted_mat.min() − λ₁·maxvar_mat.max() + λ₂·entropy_min`, com `entropy_min` = soma dos mínimos da tabela por k (0,291 no KuaiRec). Recompensa final = `clip0(r̂ − λ₁·P_U + λ₂·P_E − MIN_R)` (sem intervenção de exposição).
- **Alvo do modelo de usuário:** `watch_ratio_normed` = (watch_ratio − média do vídeo) / desvio do vídeo, limitado a 10, calculado juntando big e small (`preprocessing_kuairec.py`). O carregador `KuaiEnv.get_df_kuairec` lê só `user_id, item_id, timestamp, watch_ratio_normed, duration_normed`.
- **Recompensa de avaliação:** `KuaiEnv.load_mat` usa o `watch_ratio` bruto do small_matrix limitado a 5; células ausentes ficam 0 (17.827 no KuaiRec); NaN recebe a média.
- **Ordem de usuários e itens:** ambiente e matrizes do modelo de usuário usam `LabelEncoder` (IDs ordenados) sobre o small_matrix.
- **Treino:** 100 ambientes simulados em `DummyVectorEnv` (mesmo processo).

## 6. Observações em aberto

- No FB (repetição permitida), o MBPO dura ~27,6 passos de 30, contra 4,6 da aleatória e 2,7–3,2 dos bandits. Hipótese: a política aprende a alternar 5+ itens de categorias disjuntas, o que nunca dispara a regra com N = 4.
- Itens raros têm P_E (k=1) maior (Spearman −0,53 com popularidade). Hipótese: artefato da normalização (continuações únicas vistas uma vez valem 1).
- Os métodos de RL têm MCD (0,32–0,48) bem abaixo da taxa de base do acaso (0,83). Hipótese: evitar categorias frequentes prolonga a sessão.
- O R_each dos métodos de RL (exceto o MOPO no artigo) fica abaixo do acaso; a vantagem em R_tra vem do Length.

## 7. Como os scripts foram validados

Todos os scripts de análise foram testados contra cópias literais do código do DORL em dados sintéticos (`tests/`). A política aleatória é uma reimplementação: não passa pelo pipeline do DORL, mas reproduz trajetórias, recompensas e MCD idênticas às do ambiente e do avaliador copiados.
