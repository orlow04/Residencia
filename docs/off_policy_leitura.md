# Off-policy evaluation: o que ler para entender o código

Guia de estudo para `ope.py`, `ope_contexto.py`, `ope_dados.py` e `ope_politicas.py` (branch `off-policy`).
As referências fora do seu referencial (Sutton & Barto, Aggarwal, Afsar et al.) foram citadas de memória:
**confirme título, ano e seção antes de usar no relatório.**

## A ideia em um parágrafo

O log real foi gerado por uma política que não é a sua: a política de registro, π_β, que no KuaiRec é o sistema
da plataforma somado ao comportamento do usuário. Você quer saber quanto valeria outra política, π_e, a política-alvo
(MBPO, DORL…), sem executá-la. O *importance sampling* resolve isso reponderando cada passo do log pelo peso
w = π_e(a|x) / π_β(a|x): ações que π_e escolheria mais do que π_β contam mais, e as que escolheria menos contam
menos. Todo o resto do código é consequência de três problemas dessa ideia:

1. **Variância.** Os pesos têm variância alta quando π_e se concentra onde π_β quase não passou.
2. **π_β desconhecida.** Ela precisa ser estimada, e o erro de estimação vira viés.
3. **Suporte.** Não se avalia o que o log nunca mostrou.

## Roteiro mínimo, em ordem

| # | Leitura | Para quê | Custo |
|---|---|---|---|
| 1 | Sutton & Barto, 2ª ed., cap. 5, seção de *off-policy prediction via importance sampling* (5.5, a confirmar) | A base. *Ordinary importance sampling* é o IPS; *weighted importance sampling* é o SNIPS. A discussão de viés × variância entre os dois é a mesma que aparece nas validações | 1–2 h; já está no seu referencial |
| 2 | Dudík, Langford & Li (2011), "Doubly Robust Policy Evaluation and Learning", ICML. Seções 1 a 3 | Formula o problema como bandit contextual por passo, que é o cenário do `ope.py`. Define o modelo direto (DM), o IPS e o *doubly robust* (DR), e prova quando o DR não tem viés | 2–3 h |
| 3 | Swaminathan & Joachims (2015), "The Self-Normalized Estimator for Counterfactual Learning", NeurIPS. Seções 1 a 3 | O SNIPS e por que ele é mais estável quando os pesos estão mal calibrados. É o estimador principal escolhido aqui | 2 h |
| 4 | Chen et al. (2019), "Top-K Off-Policy Correction for a REINFORCE Recommender System", WSDM. Seções 3 e 4 | Recomendação em escala (YouTube) com π_β estimada a partir do log. Liga o OPE à sua trilha de REINFORCE e ao problema 2 | 2–3 h |

Com as leituras 1 e 2 você entende o `ope.py` inteiro. A 3 justifica a escolha do SNIPS. A 4 dá o contexto de
recomendação e de π_β estimada.

Opcionais, se sobrar tempo:
- Bottou et al. (2013), "Counterfactual Reasoning and Learning Systems", JMLR: corte dos pesos e diagnósticos.
- Gilotte et al. (2018), "Offline A/B Testing for Recommender Systems", WSDM: estimadores com corte e normalização
  em produção.
- Li et al. (2011), "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation
  Algorithms", WSDM: o caso em que π_β é uniforme e conhecida.

## Conceito → onde está no código → onde ler

| Conceito | No código | Leitura |
|---|---|---|
| Valor de uma política, V(π_e) = E[Σ_a π_e(a\|x)·r(x, a)] | cabeçalho do `ope.py` | Dudík et al., seção 2 |
| Peso de importância w = π_e / π_β | `ope.pesos` | Sutton & Barto 5.5 (*importance-sampling ratio*) |
| IPS (média de w·r) | `ope.ips` | Sutton & Barto 5.5 (*ordinary*); Dudík et al. |
| SNIPS (Σ w·r / Σ w) | `ope.snips` | Sutton & Barto 5.5 (*weighted*); Swaminathan & Joachims |
| Corte dos pesos | `ope.ips_corte` | Bottou et al. |
| Modelo direto (DM) e *doubly robust* (DR) | `ope.dm`, `ope.dr` | Dudík et al., seção 3 |
| Tamanho efetivo da amostra, ESS = (Σw)² / Σw² | `ope.ess` | Sutton & Barto não cobre; ver Owen, *Monte Carlo theory, methods and examples*, cap. 9 (a confirmar) |
| Suporte: π_β > 0 onde π_e > 0 | mistura ε nas validações; ESS e maior peso no `ope_politicas.py` | Dudík et al.; Bottou et al. |
| π_β estimada pelo log | `ope_dados.propensoes`; comparação das três versões no `ope_contexto.py` | Chen et al., seção 4 |
| Validação semi-sintética (valor verdadeiro conhecido) | `ope.py --validacao`, `ope_contexto.py` | Dudík et al., seção de experimentos (mesma lógica: dados totalmente rotulados viram um bandit) |
| Intervalo de confiança por *bootstrap* de usuários | `ope_politicas.estimativas` | Estatística geral: reamostra-se a unidade independente, que é o usuário, não o passo |
| Estado e π_e(a\|s) da política treinada | `ope_politicas.estado`, `pi_e` | Código do DORL: `state_tracker2.py:138`, `a2c2.py:150` |

## O que é resultado seu, não da literatura

Dois pontos das validações não vêm prontos de nenhuma dessas leituras, e convém saber explicá-los:

1. **O IPS fica enviesado com π_β contada no próprio log** (−0,57 a −0,80; `results/ope/ope_contexto.txt`). A tabela
   (item anterior → item) tem 3.327² células, e 23% dos passos são pares vistos uma só vez. Todo par observado tem
   contagem ≥ 1, o que superestima π_β justamente nas ações registradas e encolhe os pesos. O SNIPS divide por Σw e
   cancela o fator comum.
2. **O termo de entropia do DORL e a π_β do OPE descrevem o mesmo objeto.** O P_E de k = 1 é a entropia da
   distribuição do próximo item dado o último, no log; o OPE usa essa distribuição como política de registro. As
   tabelas não são idênticas: a do DORL usa todos os itens do big_matrix e não corta sessões
   (`get_save_entropy_mat`); a do OPE usa só os itens do small_matrix, dentro de sessões de 30 min.

## Perguntas de verificação

Depois das leituras, você deve conseguir responder sem consultar:

1. Por que o IPS não tem viés quando π_β é conhecida, e o que acontece se π_β(a|x) = 0 para uma ação que π_e escolhe?
2. Por que o SNIPS tem viés, mas costuma errar menos que o IPS? (Compare com *ordinary* × *weighted* no Sutton &
   Barto.)
3. O DR "não tem viés se π_β **ou** o modelo direto estiver correto". Na validação, o DR com π_β por contagens errou
   o oráculo por −1,98. Qual das duas condições falhou? E a outra?
4. O que um ESS de 265 em 100.000 passos diz sobre a estimativa?
5. Por que a avaliação aqui é por passo, e não por sessão inteira? O que acontece com o peso de uma sessão de 20
   passos?
6. O OPE estima o valor de escolher o próximo item dado um histórico real. Em que isso difere do que o simulador
   mede com Length e R_tra?
