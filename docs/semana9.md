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

**Resultado (verificado; `results/bolha/bolha_saida.txt`).** NX_10, 5 sementes por método, 20.000 trajetórias por
semente; média ± dp entre sementes.

| Método | V | B_V | Fração com B_V > 0 | D | B_D | Fração com B_D > 0 |
|---|---|---|---|---|---|---|
| MBPO | 9,69 ± 0,17 | −1,67 ± 0,17 | 0,025 | 0,893 ± 0,003 | −0,066 | 0,004 |
| MOPO | 9,63 ± 0,22 | −1,62 ± 0,22 | 0,032 | 0,892 ± 0,005 | −0,065 | 0,005 |
| DORL | 9,96 ± 0,09 | −1,94 ± 0,09 | 0,003 | 0,899 ± 0,001 | −0,072 | 0,000 |
| DORL-const | 10,12 ± 0,30 | −2,10 ± 0,30 | 0,003 | 0,899 ± 0,001 | −0,072 | 0,000 |
| MBPO-E5 | 9,31 ± 0,24 | −1,29 ± 0,24 | 0,101 | 0,883 ± 0,006 | −0,056 | 0,022 |
| DORL-E5 | 10,03 ± 0,14 | −2,02 ± 0,15 | 0,004 | 0,898 ± 0,001 | −0,071 | 0,000 |
| Aleatória | 8,88 ± 0,00 | −0,86 ± 0,00 | 0,295 | 0,838 ± 0,000 | −0,011 | 0,321 |

**Interpretação.**

1. Nenhum método de RL produz bolha. As trajetórias têm mais categorias que o consumo real (8,02) e que o acaso
   (8,88), e de 0,3% a 10% delas são mais estreitas que o consumo do usuário, contra 30% da aleatória. O D de
   0,89–0,90 está perto do teto de 0,90 para 10 itens de uma categoria cada.
2. DORL e DORL-const são mais diversos que MBPO e MOPO (B_V −1,94 contra −1,67, acima do dp). DORL-const = DORL e
   MOPO = MBPO, dentro do dp: a diversidade não vem da variação do P_E (mesmo padrão do E2).
3. Com repetições no histórico, todos os B ficam mais negativos; a conclusão não muda.
4. Hipótese inicial: a política aprende no treino a não repetir categoria para não disparar a regra de saída e
   mantém esse comportamento no NX_10. O E3 (etapa 5) a enfraquece: mesmo com N = 1, que só proíbe repetir a
   categoria do item imediatamente anterior, nenhum método se aproxima do consumo do usuário.
5. A métrica satura: os métodos de RL estão no teto de variedade, então ela detecta bolha, mas separa mal métodos
   que já estão no teto.

**Limitação.** O usuário simulado não tem histórico entre sessões. A definição de Michiels et al., que trata a
bolha como um processo ao longo do tempo, não pode ser testada neste simulador; mede-se apenas o estado.

## Etapa 4 — E5: modelo de usuário treinado com o sinal da avaliação

**Pergunta.** Se o modelo de usuário aprender o mesmo sinal da avaliação (watch_ratio bruto limitado a 5), em vez
do watch_ratio padronizado por vídeo (achado A4, S2), a vantagem do DORL e a ordem dos métodos mudam?

**O que foi feito.** O `e5_alvo.patch` troca o alvo em `KuaiEnv.get_df_kuairec` (`KuaiEnv.py:66`), ativado só pela
variável `E5_RAW_WR`; sem ela, a saída é idêntica à original (`tests/test_e5.py`). O `roda_e5.sh` treina o modelo
(`--message rawwr`), confere que a tabela de entropia não mudou e roda MBPO e DORL com 5 sementes cada
(`E5_{MBPO,DORL}_s*`, com trajetórias gravadas). Única mudança em relação ao R: o modelo de usuário.

**Resultado (verificado).** Escala do modelo (`results/e5/modelo_rawwr_resumo.txt`) e S2 refeita sobre ele
(`results/e5/rhat_aval_rawwr.json`):

| | pointneg (original) | rawwr (E5) |
|---|---|---|
| r̂: média / dp / mínimo | −0,002 / 0,022 / −0,453 | 0,008 / 0,013 / −0,030 |
| Bônus implícito por passo do MBPO (r̂ médio − r̂ mínimo, achado A3) | 0,45 | 0,04 |
| Correlação global r̂ × avaliação (Pearson) | 0,083 | 0,152 |
| Spearman por usuário, mediana | 0,075 | 0,181 |
| Recompensa de avaliação do top-10 do r̂ (acaso 0,87; média do vídeo 2,33) | 0,40 | 1,44 |

Métricas (`results/e5/e1_resumo.csv`), NX_0, média ± dp entre 5 sementes:

| Método | R_tra | R_each | Length | MCD |
|---|---|---|---|---|
| MBPO | 11,69 ± 2,32 | 0,736 ± 0,071 | 15,78 ± 1,83 | 0,323 ± 0,061 |
| MBPO-E5 | 12,32 ± 1,50 | 1,034 ± 0,152 | 12,07 ± 1,52 | 0,488 ± 0,056 |
| DORL | 18,27 ± 2,46 | 0,780 ± 0,086 | 23,42 ± 1,10 | 0,431 ± 0,027 |
| DORL-E5 | 17,73 ± 4,36 | 0,777 ± 0,078 | 22,44 ± 4,49 | 0,406 ± 0,034 |
| Aleatória | 4,01 ± 0,02 | 0,872 ± 0,001 | 4,60 ± 0,02 | 0,827 ± 0,001 |

No NX_10, o R_each do MBPO-E5 é 1,049 ± 0,197 (MBPO: 0,708). No FB, o seu Length cai de 27,6 para 5,6 e a MCD vai a
0,81, perto da aleatória.

**Interpretação.**

1. O r̂ do E5 ordena os itens melhor que o original e melhor que o acaso (top-10 1,44 contra 0,40 e 0,87), mas
   ainda bem abaixo da média do vídeo (2,33). O alinhamento do sinal foi parcial.
2. O MBPO muda: R_each +0,30 (acima do dp), Length −3,7 e MCD +0,17. É o primeiro método de RL com R_each acima
   do acaso, por cerca de um dp. A rotação de categorias no FB desaparece.
3. O DORL não muda: DORL-E5 = DORL nas quatro métricas e nos três protocolos, e também na bolha.
4. O E5 mudou duas coisas: a ordenação do r̂ e o bônus implícito por passo do MBPO, que caiu cerca de 12 vezes
   (de cerca de 20 para 3 vezes o dp do r̂). Hipótese coerente com o E2: o comportamento é decidido pela razão
   entre o bônus por passo e a variação do r̂. No MBPO, o bônus encolheu e a política passou a escolher itens; no
   DORL, o bônus é dominado por λ₂·(P_E − 0,291) ≈ 8,35, que não mudou, e o r̂ continua irrelevante. O desenho
   não separa a contribuição da ordenação melhor da do bônus menor.

**Limitações.** Um único treino do modelo de usuário; a matriz salva não está na escala do alvo (o treino ajusta
y/(1 + score) e empurra negativos para 0; `run_worldModel_ensemble.py:385` e `:415`).

## Etapa 5 — E3: sensibilidade ao N da regra de saída

**Pergunta.** A vantagem do DORL e a diversidade das trajetórias dependem do tamanho da janela da regra de saída?

**O que foi feito.** MBPO e DORL treinados e avaliados com N = 1 e N = 8 (o R usa N = 4), 5 sementes cada, com as
trajetórias gravadas (`roda_e3.sh`, mensagens `E3N{1,8}_{MBPO,DORL}_s*`). O `--num_leave_compute` vai para o
`KuaiEnv` (`src/core/configs.py:170`), e o ambiente simulado do treino usa o `done` desse mesmo ambiente
(`src/core/worldModel/simulated_env.py:165`): o N muda no treino e na avaliação. O script confere o N registrado em
cada log. A política aleatória foi refeita com o mesmo N (`results/e3/aleatoria_N{1,8}.json`).

**Resultado (verificado; `results/e3/e1_saida.txt`, `results/bolha/bolha_saida.txt`).** NX_0, média ± dp entre
5 sementes; V da bolha sob NX_10.

| N | Método | R_tra | R_each | Length | MCD | V (NX_10) |
|---|---|---|---|---|---|---|
| 1 | MBPO | 16,69 ± 1,90 | 0,740 ± 0,067 | 22,62 ± 2,01 | 0,375 | 9,35 ± 0,24 |
| 1 | DORL | 21,40 ± 2,30 | 0,780 ± 0,076 | 27,40 ± 0,46 | 0,448 | 10,23 ± 0,37 |
| 1 | Aleatória | 6,64 ± 0,04 | 0,872 ± 0,002 | 7,61 ± 0,05 | 0,828 | 8,88 |
| 4 | MBPO | 11,69 ± 2,32 | 0,736 ± 0,071 | 15,78 ± 1,83 | 0,323 | 9,69 ± 0,17 |
| 4 | DORL | 18,27 ± 2,46 | 0,780 ± 0,086 | 23,42 ± 1,10 | 0,431 | 9,96 ± 0,09 |
| 4 | Aleatória | 4,01 ± 0,02 | 0,872 ± 0,001 | 4,60 ± 0,02 | 0,827 | 8,88 |
| 8 | MBPO | 13,49 ± 2,83 | 0,690 ± 0,078 | 19,45 ± 2,28 | 0,316 | 9,90 ± 0,01 |
| 8 | DORL | 15,79 ± 0,71 | 0,789 ± 0,033 | 20,08 ± 0,85 | 0,410 | 10,10 ± 0,35 |
| 8 | Aleatória | 3,80 ± 0,02 | 0,871 ± 0,002 | 4,36 ± 0,02 | 0,828 | 8,88 |

(A aleatória da bolha não depende do N: no NX_10 a regra está desligada.)

**Interpretação.**

1. **A afirmação principal depende do N.** DORL − MBPO em Length: +4,8 com N = 1 e +7,6 com N = 4, acima do dp;
   +0,6 com N = 8, dentro do dp (2,3). Em R_tra, +2,3 com N = 8, dentro do dp (2,8). Com a janela maior, a
   vantagem do DORL desaparece.
2. **O Length não é monótono no N para o MBPO** (22,6 → 15,8 → 19,4), embora a regra fique mais restritiva.
   Hipótese: o recorte da janela (A5) muda com o N; com N = 8, a janela encolhe nos passos 5 a 7. A aleatória
   quase não muda de N = 4 para N = 8 (4,60 → 4,36).
3. **R_each e NX_10 não mudam:** em todos os N, os métodos de RL ficam abaixo do acaso em R_each (0,69–0,79
   contra 0,87) e em R_tra no NX_10. A MCD fica entre 0,32 e 0,45, longe dos 0,83.
4. **A bolha não aparece com nenhum N.** O V do MBPO acompanha o N (9,35 → 9,69 → 9,90, diferenças acima do dp),
   mas mesmo com N = 1 fica acima do consumo do usuário (8,02) e do acaso (8,88). O DORL fica no teto em todos os N.
   A regra de saída explica só uma parte pequena da diversidade. Hipótese alternativa: a diversidade é o outro lado
   da MCD baixa; evitar as 10 categorias dominantes empurra a política para a cauda de categorias e aumenta o V.
5. **O FB mantém a rotação:** o MBPO dura cerca de 25 passos com N = 1 e com N = 8 (27,6 com N = 4).

**Limitações.** Dois valores de N; MOPO e DORL-const não foram repetidos; o recorte A5 está presente em todos os N.

## Reprodutibilidade e organização

| Item | Arquivo |
|---|---|
| Prova de que o E2 usou `PE_CONST=1.9609` (o histórico do shell foi perdido): 2 épocas por semente repetidas, as 5 idênticas e o controle diferente | `verifica_E2.sh` → `results/pe/verifica_E2.txt`; comandos em `results/pe/comandos_P_E2.md` |
| Índice dos resultados, com os logs descartados e o motivo de cada um | `results/README.md` |
| Resumo no formato da Tabela 2 de todas as execuções, nos três protocolos | `results/table2/` |
| Figuras regeneradas a partir de `results/` | `figuras.py` → `docs/figuras/` |

## Pendências

- Teste da hipótese alternativa da bolha: relação, por trajetória, entre V e a fração de itens de categorias
  dominantes; e fração dos passos NX_10 que teriam disparado a regra de saída.
- Investigação do FB com as trajetórias gravadas.
