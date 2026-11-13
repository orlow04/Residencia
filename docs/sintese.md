# Síntese dos experimentos: afirmação × sinal × evidência × força

Estado em 11/11/2026. Versão para artigo (LaTeX, `booktabs`): `docs/tabelas/sintese.tex`.
Cada número vem de um arquivo em `results/`, indicado na coluna de evidência ou no registro da etapa
(`docs/experimentos_semana8.md`, `docs/semana9.md`). Comparações entre métodos: média ± dp entre 5 sementes; uma
diferença conta quando supera o maior dp entre sementes dos dois métodos.

**Sinal** (eixos da pergunta orientadora): T = sinal de treino (r̂, P_U, P_E); S = regra do simulador (regra de saída);
M = métrica ou protocolo de avaliação.

**Controle** que produziu a evidência: (a) desvio entre sementes; (b) referência sem sinal (política aleatória);
(c) ablação por constante; (d) variação da regra de saída ou do protocolo; (e) concentração medida em vários níveis;
(f) referência empírica (comportamento real no big_matrix).

**Força**: *Reproduzida* (diferença acima do dp, na direção do artigo); *Condicional* (reproduzida só sob uma
condição do simulador); *Dentro do ruído* (diferença menor que o dp); *Não sustentada* (o efeito aparece sem o
mecanismo alegado); *Invertida* (outro nível ou protocolo dá a ordem oposta); *Não testada*.

## Tabela 1 — Afirmações do artigo (Gao et al., SIGIR '23)

| # | Afirmação | Sinal | Controle | Evidência | Força |
|---|---|---|---|---|---|
| A1 | O DORL prolonga a sessão: supera o MBPO em Length e R_tra (Tabela 2) | S, M | a, d | NX_0: +7,6 em Length e +6,6 em R_tra, acima do dp (R). Com o comprimento fixado (NX_10), os 4 métodos de RL ficam abaixo do acaso em R_tra, 7,08–7,39 contra 8,72 (E1). N = 1/4c/8c: +4,8/+6,7/+3,5 (E3/A5); com N = 4c/8c, todo o ganho vem do passo MOPO → DORL-const (+5,7/+3,0, semana 10). Saída calibrada: +0,3 passo, RL ≈ acaso (E6) | Condicional (só com a regra determinística) |
| A2 | C1: a penalidade de incerteza produz efeito Mateus (MCD do MOPO > MBPO) | T, M | a, e | MCD +0,043, contra um dp de 0,085 (R). Saída calibrada: MOPO-E6 = MBPO-E6 em cobertura (50,2% contra 52,8%) e MCD (0,669 contra 0,684) (E6) | Dentro do ruído |
| A3 | C4: o termo de entropia reintroduz exploração (diversidade) | T | c | P_E constante = DORL nas 4 métricas (E2), com a janela corrigida N = 4c/8c (Length +0,8/+0,9, dentro do dp; semana 10) e na cobertura sob a saída calibrada (2,5% contra 2,7%, E6). O P_E da política treinada é 1,96 de 2, quase constante (P) | Não sustentada (o efeito vem do *survival bonus*) |
| A4 | O DORL reduz a concentração (MCD comparável ou menor) | M | b, e | Todos os métodos de RL ficam abaixo da taxa de base do acaso, 0,83 (S3). Cobertura de itens: DORL 3,3%, MBPO 12%, aleatória 86,5% (etapa 8); E6: 2,7% contra 52,8%, personalização 0,15 | Invertida no nível do item |
| A5 | O RL supera os bandits (Tabela 2) | M | a, b, d | Com 5 sementes: em NX_0 o RL supera os bandits em R_tra e Length; em R_each o UCB supera MBPO, MOPO e DORL (0,869 contra 0,74–0,78). Em NX_10, o UCB supera MBPO e MOPO em R_tra (8,45 contra 7,08/7,39), acima do dp; contra DORL e DORL-const, limítrofe ou ruído. Os bandits ficam abaixo do acaso em R_tra e Length nos 3 protocolos (semana 10, etapa 3) | Condicional (vale para R_tra e Length em NX_0) |
| A6 | Os números da Tabela 2 representam cada método | M | a | O ± da Tabela 2 é entre épocas de uma execução. O dp da MCD entre sementes (0,061) é 3,6× a diferença DORL − MBPO (0,017). A semente 2 é a única a menos de 1 dp do artigo nas 4 métricas (R) | Não sustentada com uma semente |
| A7 | C2 (repetição prejudica o longo prazo) e C3 (a causa é suprimir amostras incertas) | — | — | C2 é correlacional, em dados reais da plataforma; C3 é passo teórico. Fora do alcance do simulador | Não testada |

## Tabela 2 — Achados deste trabalho sobre os sinais

| # | Achado | Sinal | Controle | Evidência | Força |
|---|---|---|---|---|---|
| B1 | O sinal de treino (r̂) não ordena os itens como a avaliação | T, M | b | Top-10 do r̂: 0,40, contra 0,87 do acaso e 2,33 da média do vídeo (S2). Com o alvo da avaliação, o MBPO muda (R_each 0,736 → 1,034) e o DORL não (E5) | Verificada (estática e com treino) |
| B2 | O P_E funciona como *survival bonus* (bônus fixo por passo) | T | c | O P_E responde por ~99% da variação de r̃ nas políticas de referência (S1); 1,96 de 2 na política treinada (P); E2 | Verificada |
| B3 | A política explora a regra de saída | S | d | No FB, o MBPO cicla com bloco de N + 1 itens em 80–99% dos ciclos puros (N = 1, 4). A vantagem varia com N e com a janela; o N = 8 original era artefato do recorte (+0,6). MOPO cicla como o MBPO e DORL-const como o DORL (ciclo puro 0,60 contra 0,12, N = 4c): o ciclo depende da ausência do bônus | Verificada |
| B4 | A regra de saída não corresponde ao usuário real | S | d, f | P(fim \| repete categoria) = 6,3% contra 2,6% nas sessões reais; o simulador supõe 100% contra 0% (etapa 9) | Verificada (associação, não efeito causal) |
| B5 | O comportamento é decidido pela razão entre o *survival bonus* e a variação do r̂ | T, S | c, d | Bônus implícito do MBPO 0,45 → 0,04 no E5, e o MBPO passa a escolher itens. Um termo por vez: só o *survival bonus* derruba a cobertura, na saída calibrada (50,2% → 2,5%, E6) e com a janela corrigida (10,4% → 3,3% em N = 4c; 5,7% → 3,8% em N = 8c); P_U e variação do P_E, nada | Hipótese apoiada (tamanho do bônus não variado) |
| B6 | A concentração depende do nível medido | M | b, e | MCD (categorias) × cobertura (itens) dão ordens opostas; nenhum método forma bolha individual: V 9,2–11,4 contra 8,02 do usuário e 8,88 do acaso (bolha, etapa 8) | Verificada |
| B7 | A repetição permitida no FB infla a concentração e não segue a popularidade, ao contrário da revisita real | S, M | e, f | As 18 variantes repetem 6,7–74,3% dos passos FB, contra 3,4% nas sessões reais de 10–30 itens. Cobertura com e sem repetições: MBPO 3,9% / 6,6%, MBPO-N1 6,5% / 16,9%, DORL-E6 0,9% / 2,3%. Revisitas reais vão ao top 10% de popularidade 2× mais que as visualizações (20,3% contra 10,6%); nas políticas, passos repetidos ≈ passos em geral (MBPO 17,3% contra 17,8%), abaixo dos 30,6% do acaso (semana 10, revisitas) | Verificada (dp de popularidade grande) |

## Notas

- Semana 10 concluída: bandits com 5 sementes (A5, 13/11); revisitas (B7, 11/11); janela corrigida para
  MOPO e DORL-const (A1, A3, B3, B5; 12/11).
- Uma calibração da saída probabilística, um conjunto de dados (KuaiRec), λ₁ e λ₂ fixos nos valores do artigo.
