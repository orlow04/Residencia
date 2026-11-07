# Semana 9 — Da concentração agregada à bolha por usuário

Registro técnico da semana. Referências de código: DORL-codes no commit `818c9bd` (caminhos relativos a
`DORL-codes/`); arquivos deste repositório na raiz.

## Motivação

Dois resultados da semana 8 apontaram o problema:

- A MCD é calculada sobre os itens de **todas** as trajetórias de uma época, achatados numa só lista
  (`Callback_Coverage_Count.get_actions_feat`, `src/core/evaluation/evaluator.py:265`). Ela não distingue um
  usuário que recebeu 10 itens da mesma categoria de dez usuários que receberam uma categoria diferente cada.
- A S3 mostrou que a MCD só tem significado diante de uma referência (0,83 do acaso, contra 0,32–0,43 dos
  métodos de RL).

A pergunta do planejamento era em que ponto do código uma recomendação é considerada boa e se o que as métricas
chamam de concentração corresponde ao que a literatura chama de bolha. Para medir uma bolha **por usuário**,
são necessárias duas coisas que o código não fornece:

1. **O que cada usuário recebeu.** O avaliador só grava médias por época (etapa 2).
2. **Uma referência de diversidade para cada usuário.** Ela não pode vir do simulador. A etapa 1 mostra que o
   usuário simulado não tem nenhuma noção de diversidade além da regra de saída, então a referência tem de vir
   do consumo real.

## Etapa 1 — O que o usuário simulado sabe

Leitura do código para saber quais sinais definem o "usuário" no KuaiRec.

| Componente | Sinais usados | Onde no código |
|---|---|---|
| Modelo de usuário (r̂) | Entrada: `user_id`, `item_id`, até 4 categorias do item, duração do vídeo. Alvo: `watch_ratio_normed` | `src/core/configs.py:18-21` (`get_features`, ramo `KuaiEnv-v0`); `environments/KuaiRec/env/KuaiEnv.py:66` (`get_df_kuairec`) |
| Características do usuário | **Não usadas**: o ramo do KuaiRec fixa `user_features = ["user_id"]`; os logs dos autores registram `is_userinfo: false` | `src/core/configs.py:18-21`; `run_Policy_Main.py:62`; `results/logs/[pointneg]_*.log` |
| Termo de entropia (P_E) | Só a ordem dos itens de cada usuário no log | `src/core/user_model_ensemble.py` (`get_save_entropy_mat`) |
| Regra de saída | Só as categorias dos últimos N itens | `environments/KuaiRec/env/KuaiEnv.py:344` (`_determine_whether_to_leave`) |
| Recompensa de avaliação | `watch_ratio` bruto do small_matrix, limitado a 5 | `environments/KuaiRec/env/KuaiEnv.py:223` (`load_mat`) |

**Conclusão.** O único sinal de comportamento é o watch_ratio (tempo assistido ÷ duração). Compartilhamento e
curtida não existem por interação no KuaiRec. Rever o vídeo aparece só indiretamente: watch_ratio > 1, e 11,8%
das linhas do big_matrix dos 1.411 usuários de avaliação repetem o par usuário–vídeo. No simulador, a diversidade
só importa pela regra de saída, que encerra a sessão quando uma categoria se repete. Por isso, a referência de
diversidade de cada usuário vem do seu histórico real (etapa 3).

## Etapa 2 — Registro das trajetórias de avaliação

**O que foi feito.** Um patch no avaliador grava, a cada passo de avaliação, a época, o protocolo, a trajetória,
o passo, o usuário, o item (codificado e original) e a recompensa. O gancho fica logo após a caminhada que
calcula a MCD (`evaluator.py:320`) e percorre o buffer da mesma forma, portanto grava exatamente os itens que
entram na MCD. O patch só é ativado pela variável `TRAJ_LOG`.

**Validação.**

| Nível | O que verifica | Arquivo |
|---|---|---|
| Teste sintético | O CSV gravado é igual à caminhada de `get_actions_feat` (cópia literal), com o `VectorReplayBuffer` real do tianshou; sem a variável, o gancho não roda | `tests/test_traj.py` |
| Execução real (2 épocas) | (1) Patch inerte: épocas idênticas ao log original; (2) R_tra, Length, R_each e MCD recalculados do CSV iguais aos do log, por época e protocolo; (3) recompensa gravada = matriz de avaliação | `verifica_traj.sh` → `results/trajetorias/verifica_traj.txt` |
| Execuções completas | As 20 políticas (MBPO, MOPO, DORL e DORL-const × 5 sementes) reexecutadas com o registro; cada uma conferida automaticamente como idêntica ao log original nas quatro métricas, nas 200 épocas | `roda_trajetorias.sh` |

Não havia políticas salvas (`is_save` desligado em todos os logs). A reexecução só é válida porque o treino é
determinístico: o `DORL_pelog_s2` já reproduzia o `DORL_s2`, e cada reexecução desta semana reproduziu o seu
original.

## Etapa 3 — Definição operacional de bolha

**Base na literatura.** Michiels et al. (RecSys 2023, "How Should We Measure Filter Bubbles?") definem a bolha
tecnológica como "uma diminuição da diversidade das recomendações de um usuário ao longo do tempo". Eles medem a
**variedade**, isto é, o número de tópicos distintos, por usuário e por semana, e a regridem no tempo desde o
cadastro com um GLMM binomial negativo (código em `github.com/LienM/recsys-23`). Usam a taxonomia de Stirling:
variedade, equilíbrio e disparidade.

**Adaptação ao simulador.**

| Escolha | Decisão | Motivo |
|---|---|---|
| Protocolo | NX_10 | Todas as trajetórias têm 10 itens; a contagem de categorias não depende do comprimento |
| Variedade V | Categorias distintas na união das categorias dos 10 itens | A medida de Michiels et al., adaptada a itens com 1 a 4 categorias |
| Equilíbrio D | Gini-Simpson 1 − Σ q_c²; item com k categorias conta 1/k para cada uma | Distingue "9 itens de A + 1 de B" de "5 + 5", que têm a mesma variedade |
| Referência do usuário | Média de V e D nas janelas de 10 itens consecutivos do histórico no big_matrix (dado de treino), sem repetir itens | Mesma unidade da trajetória; o NX_10 também não repete itens |
| Índice | B = referência − trajetória; B > 0 indica trajetória mais estreita que o consumo real | Bolha relativa ao próprio usuário |
| Segunda referência | Política aleatória (usuário e 10 itens distintos uniformes) | Separa o efeito do catálogo do efeito da política |

Implementação: `bolha.py`; teste contra a definição calculada item a item: `tests/test_bolha.py`.

**Resultado parcial (verificado).**

| | V | D |
|---|---|---|
| Consumo real do usuário (janelas de 10 itens) | 8,02 | 0,827 |
| Política aleatória, NX_10 (5 sementes) | 8,88 | 0,838 |

O consumo real já é mais concentrado que o acaso. Uma bolha produzida pela política tem de ficar abaixo do
consumo real do usuário, não apenas abaixo do acaso. Com repetições no histórico, a referência fica ainda mais
estreita (B_V da aleatória = −1,29 contra −0,86); excluir as repetições é a escolha mais conservadora.

**Métodos de RL:** [a preencher com `results/bolha/bolha_saida.txt`].

**Limitação.** O usuário simulado não tem histórico entre sessões. A definição de Michiels et al., que trata a
bolha como um processo ao longo do tempo, não pode ser testada neste simulador; mede-se apenas o estado.

## Reprodutibilidade e organização

| Item | Arquivo |
|---|---|
| Prova de que o E2 usou `PE_CONST=1.9609` (o histórico do shell foi perdido): 2 épocas por semente repetidas, as 5 idênticas e o controle diferente | `verifica_E2.sh` → `results/pe/verifica_E2.txt`; comandos em `results/pe/comandos_P_E2.md` |
| Índice dos resultados, com os logs descartados e o motivo de cada um | `results/README.md` |
| Resumo no formato da Tabela 2 de todas as execuções, nos três protocolos | `results/table2/` |
| Figuras regeneradas a partir de `results/` | `figuras.py` → `docs/figuras/` |

## Pendências

- Resultado da bolha para MBPO, MOPO, DORL e DORL-const.
- E5 (modelo de usuário com watch_ratio bruto) e E3 (N da regra de saída): [preencher conforme o andamento].
- Investigação do FB com as trajetórias gravadas.
