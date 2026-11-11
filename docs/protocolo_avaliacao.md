# Protocolo de avaliação de sistemas de recomendação baseados em reforço (versão 0)

Documento técnico de trabalho, iniciado em 11/11/2026. Formaliza o processo de avaliação usado nas semanas 8 a 10 para
que a pesquisa possa continuar a partir dele. Não é texto do relatório final.

Ponto de partida: a pergunta orientadora, **quais sinais usados são influentes para avaliar um sistema de recomendação
baseado em reforço?** O protocolo responde na forma de procedimento: para cada objetivo de um RLRS, que sinal o
influencia, como medi-lo, contra que referência e com que controle.

## 1. Princípio

Uma afirmação sobre um RLRS avaliado em simulador ("o método X melhora o objetivo Y") só é aceita quando a diferença
sobrevive a quatro condições:

1. **Supera o ruído de semente.** Ver o controle (a).
2. **É lida contra uma referência sem sinal e contra o comportamento real.** Ver os controles (b) e (f).
3. **Não aparece sem o mecanismo alegado.** Ver o controle (c).
4. **Não depende de uma regra arbitrária do simulador ou do protocolo.** Ver os controles (d) e (e).

No DORL/KuaiRec, cada afirmação do artigo falhou em pelo menos uma delas (`docs/sintese.md`, Tabela 1).

## 2. Componentes a separar

Na estrutura de RLRS de Afsar et al. (agente, ambiente, estado, ação, recompensa), a avaliação em simulador mistura três
fontes de sinal. O protocolo exige que cada resultado seja atribuído a uma delas.

| Código | Fonte | No DORL/KuaiRec | Por que influencia a avaliação |
|---|---|---|---|
| T | Sinal de treino: o que a política maximiza | r̂ (DeepFM), P_U, P_E | A política otimiza T, não o que a avaliação mede (S2, E5) |
| S | Regra do simulador: o que encerra ou altera a sessão | Saída ao repetir categoria (N, janela), máximo de 30 passos | A política aprende a explorar S (FB, E3/A5, E6) |
| M | Métrica e protocolo: o que se mede e em que condição | R_tra, R_each, Length, MCD; FB, NX_0, NX_10 | A ordem dos métodos muda com M (E1, etapa 8) |

## 3. Objetivos de um RLRS e como avaliá-los

Os objetivos seguem os objetivos gerais do desenho de avaliação de Aggarwal (acurácia, cobertura, novidade, diversidade,
robustez), acrescidos dos dois objetivos próprios do RL: valor de longo prazo e plausibilidade do ambiente.

| # | Objetivo | Pergunta | Métrica mínima | Referências obrigatórias | Controles | Falha observada no DORL/KuaiRec |
|---|---|---|---|---|---|---|
| O1 | Satisfação imediata | A política escolhe itens melhores a cada passo? | R_each | Aleatória; média do item (sem personalização); oráculo | a, b | RL abaixo do acaso nos 3 protocolos: 0,69–0,78 contra 0,87 (E1) |
| O2 | Valor de longo prazo (sessão) | A política mantém o usuário por mais tempo com itens bons? | Length e R_tra, sempre junto de R_each e de um protocolo com comprimento fixo (NX_10) | Aleatória com a mesma regra de saída | a, b, c, d | Vantagem só com a regra determinística: some com a saída calibrada (+0,3 passo, E6) e vira desvantagem em NX_10 (E1) |
| O3 | Diversidade e concentração | A exposição se concentra em poucos itens, categorias ou perfis? | Concentração em 4 níveis: categoria (MCD), item (cobertura, Gini da exposição), usuário (personalização; variedade V relativa ao próprio usuário), sessão (repetição) | Taxa de base do acaso em cada nível; janelas do próprio usuário | a, b, e | MCD e cobertura dão ordens opostas (etapa 8); DORL-E6 cobre 2,7% dos itens contra 52,8% do MBPO-E6 |
| O4 | Popularidade (efeito Mateus) | A política reforça o que já é popular? | Fração da exposição no top 10% de popularidade do log; Gini por item | Acaso (30,6%); revisitas reais (20,3% contra 10,6%) | b, f | O RL concentra-se em itens pouco populares: homogeneização, não Mateus (semana 10, etapa 1) |
| O5 | Fidelidade do sinal de treino | O que a política maximiza prevê o que a avaliação mede? | Top-k do r̂ avaliado pela recompensa de avaliação; correlação por usuário | Acaso; média do item; oráculo | b, c | Top-10 do r̂ 0,40 contra 0,87 do acaso (S2); P_E ≈ bônus fixo (S1, P, E2) |
| O6 | Plausibilidade do ambiente | O usuário simulado se comporta como o real nos aspectos que a política explora? | P(fim \| evento) do simulador contra sessões reais; repetição na sessão | Sessões reais do log (corte de 30 min, sem duplicatas) | f | Saída: 100% contra 6,3% real (etapa 9); repetição: 7–74% contra 3,4% real (semana 10) |

Regra de leitura para O2: um ganho em Length ou R_tra que não aparece em R_each nem em NX_10 é ganho de **duração**, e
deve ser reportado como tal, com a regra de saída que o produz.

## 4. Controles

Cada controle tem um alvo, um procedimento mínimo e uma regra de decisão. As letras são as da síntese.

| | Controle | O que detecta | Procedimento mínimo | Regra de decisão | Base |
|---|---|---|---|---|---|
| a | Desvio entre sementes | Diferença que é ruído de execução | ≥ 5 sementes por método; média ± dp entre sementes (não o ± entre épocas) | A diferença conta se supera o maior dp entre sementes dos dois métodos | Sutton & Barto: média de várias execuções |
| b | Referência sem sinal | Métrica sem escala; efeito produzido pelo ambiente | Política uniforme com o mesmo ambiente, regra e protocolo | Toda métrica é reportada ao lado do valor do acaso | Afsar et al.: dependência de simuladores |
| c | Ablação por constante | Termo de recompensa que age por seu valor médio, não por sua variação | Substituir o termo pela sua média na política treinada, sem mudar o deslocamento | Se o método ablado = o original, o mecanismo alegado não se sustenta | Sutton & Barto: papel da recompensa |
| d | Variação da regra e do protocolo | Resultado que depende de uma regra arbitrária | Variar N, a janela e o tipo de saída (determinística × calibrada); avaliar em FB, NX_0 e NX_10 | Uma afirmação que inverte entre condições é condicional, e a condição é reportada | Aggarwal: desenho da avaliação offline |
| e | Concentração em vários níveis | Leitura de concentração que muda com o nível | Medir categoria, item, usuário e sessão, todos contra a taxa de base | Uma afirmação de diversidade vale só para o nível medido | Aggarwal: limitações das medidas |
| f | Referência empírica | Simulador que premia comportamento irreal | Medir no log real o evento que a política explora (saída, repetição), sem duplicatas | Se a política depende de um evento implausível, o resultado é do simulador | Afsar et al.: lacuna simulador–usuário |

## 5. Sequência de avaliação

Ordem recomendada para avaliar um novo método ou repetir esta avaliação em outro conjunto de dados.

1. **Fixar o pipeline.**
   - Commits do código e das dependências.
   - Argumentos tirados dos logs dos autores, não do README.
   - Logs completos, um por semente.
2. **Ler o código antes de medir.**
   - Localizar a regra de saída e os protocolos.
   - Localizar o deslocamento da recompensa: bônus por passo implícito?
   - Comparar o alvo de treino com a recompensa de avaliação.
   - Registrar cada achado com arquivo e linha.
3. **Diagnóstico sem treino** (barato, na CPU):
   - política aleatória (b);
   - escala e variação de cada termo de recompensa;
   - fidelidade do sinal de treino (O5);
   - plausibilidade da regra de saída (f).
4. **Reprodução com sementes (a).** Comparar com o artigo pelo desvio do artigo e, entre métodos, pelo desvio entre
   sementes.
5. **Ablações dos termos de recompensa (c).** Um termo por vez.
6. **Variação da regra e do protocolo (d).**
   - Regra de saída corrigida e calibrada.
   - Os três protocolos.
7. **Registro de trajetórias.** A partir delas: concentração em vários níveis (e), repetição e popularidade (O4).
8. **Síntese.** Cada afirmação recebe sinal, controle, evidência e força, numa tabela como a `docs/sintese.md`.

## 6. Relato mínimo

Toda tabela de resultados deve trazer:

- n de sementes e média ± dp entre sementes;
- a política aleatória na mesma condição;
- R_each sempre que houver R_tra ou Length;
- a regra de saída e o protocolo usados;
- a concentração com o nível declarado e a taxa de base daquele nível;
- o arquivo de onde vem cada número.

## 7. O que o protocolo ainda não cobre (continuidade)

| Lacuna | Por que importa | Primeiro passo |
|---|---|---|
| Avaliação off-policy (IPS, doubly robust) | Avalia a política no log real, sem simulador; testa O1 e O2 sem a regra de saída | Estimar o valor das políticas treinadas com IPS sobre o big_matrix |
| Retenção entre sessões (C2 do artigo) | O objetivo de longo prazo do RL vai além de uma sessão | Usar as revisitas em outra sessão (70% das repetições reais) como sinal |
| Varredura de λ₁, λ₂ e do bônus fixo | B5 é hipótese apoiada com um único valor do bônus | Variar a constante do DORL-const sob o E6 |
| Segundo conjunto de dados (KuaiRand) | Todos os achados são de um simulador e um conjunto de dados | Repetir os passos 3 e 4 da seção 5 |
| Calibração da saída por usuário | O E6 usa duas probabilidades globais | Estimar P(fim \| repete categoria) por usuário |

## 8. Referências e pendências

- Aggarwal, *Recommender Systems: The Textbook*, capítulo de avaliação. Os nomes das seções foram dados de memória
  ("objetivos gerais do desenho de avaliação", "desenho da avaliação offline", "limitações das medidas de avaliação"):
  **confirmar no livro**.
- Sutton & Barto, *Reinforcement Learning: An Introduction*. A média de várias execuções vem do testbed de bandits
  (cap. 2), e o papel da recompensa, da hipótese da recompensa (cap. 3). **Confirmar os capítulos.**
- Afsar et al. (2022, ACM CSUR). A dependência de simuladores aparece entre os desafios de avaliação da revisão.
  **Confirmar a seção.**
- Resultados citados: `docs/experimentos_semana8.md`, `docs/semana9.md`, `docs/semana10.md`, `docs/sintese.md`.
