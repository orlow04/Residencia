# Semana 10 — resultados (anexo técnico do Termo)

17/11/2026. Reúne os números por trás da Entrega do Termo da semana 10. O registro completo, com procedimentos,
ressalvas e correções, está em `docs/semana9.md` (etapa 10), `docs/semana10.md` e `docs/off_policy.md`; cada seção
indica os arquivos de `results/` de onde vêm os números.

**Critério.** Média ± desvio padrão (dp) entre sementes. Uma diferença entre métodos conta quando supera o maior dp
entre sementes dos dois métodos; abaixo disso, fica "dentro do ruído".

## Termos usados

| Termo | Significado |
|---|---|
| MBPO, MOPO, DORL | Métodos de RL do artigo: sem penalidades; com penalidade de incerteza (λ₁·P_U); com incerteza e termo de entropia (λ₂·P_E) |
| DORL-const | DORL com o P_E substituído por uma constante: mantém só o bônus fixo por passo (ablação da semana 8) |
| *Survival bonus* (b) | Bônus fixo por passo que o termo de entropia soma à recompensa; b = λ₂·(c − 0,291), 8,35 no DORL-const |
| Regra N4 | Regra de saída original: o usuário simulado sai quando o item repete a categoria de um dos 4 anteriores (com o recorte da janela nos primeiros passos) |
| N = 4c, 8c | A mesma regra com a janela corrigida, sem o recorte |
| E6 | Saída probabilística calibrada nas sessões reais: o usuário sai com probabilidade 6,3% se o item repete a categoria do anterior e 2,6% se não repete |
| FB, NX_0, NX_10 | Protocolos de avaliação: repetição de itens permitida; repetição proibida (o da Tabela 2 do artigo); repetição proibida e sessão fixa em 10 passos |
| Length, R_tra, R_each | Comprimento da sessão; recompensa por trajetória; recompensa por passo (R_tra ÷ Length) |
| Cobertura | Itens distintos ÷ itens recomendados numa época de 100 trajetórias NX_10 (aleatória: 86,5%) |
| Personalização | 1 − Jaccard médio entre as listas de pares de trajetórias da mesma época (1 = listas sem itens em comum) |

## 1. Saída probabilística calibrada (E6)

**Por quê.** A regra de saída do simulador é determinística: o usuário sai sempre que repete categoria. Nas sessões
reais, a sessão termina em 6,3% das repetições de categoria. O E6 testa quais conclusões sobrevivem a uma saída
calibrada no comportamento real.

**O que foi feito.** MBPO, MOPO, DORL e DORL-const com a regra E6, 5 sementes cada, e a política aleatória com a
mesma regra (`e6_saida_prob.patch`, `roda_e6.sh`).

**Resultado** (`results/e6/e1_saida.txt`, `results/alem/alem_saida.txt`; `docs/semana9.md`, etapa 10).

| Método | Length (NX_0) | R_tra (NX_0) | R_each (NX_0) | Cobertura | Personalização |
|---|---|---|---|---|---|
| MBPO-E6 | 20,69 ± 0,10 | 17,33 ± 0,49 | 0,838 ± 0,027 | 52,8 ± 7,7% | 0,95 |
| MOPO-E6 | 20,76 ± 0,09 | 17,48 ± 0,15 | 0,842 ± 0,008 | 50,2 ± 5,0% | 0,96 |
| DORL-E6 | 20,98 ± 0,16 | 18,25 ± 1,51 | 0,870 ± 0,070 | 2,7 ± 0,8% | 0,15 |
| DORL-const-E6 | 20,91 ± 0,23 | 18,06 ± 0,62 | 0,864 ± 0,032 | 2,5 ± 0,5% | 0,16 |
| Aleatória | 20,10 ± 0,06 | 17,53 ± 0,07 | 0,872 ± 0,002 | 86,5% | 1,00 |

Conclusões:
- A vantagem do DORL em duração some: +0,3 passo sobre o MBPO, contra +4,8 na regra determinística de mesmo N (N = 1).
- Nenhum método de RL supera a aleatória em recompensa. Em R_tra, todos ficam dentro do dp dela; em R_each, ficam
  iguais (DORL, DORL-const) ou abaixo (MBPO, MOPO).
- A concentração do DORL sobrevive. Decomposição da cobertura:
  - MBPO → MOPO (penalidade de incerteza): 52,8% → 50,2%, dentro do ruído;
  - MOPO → DORL-const: 50,2% → 2,5%;
  - DORL-const → DORL (variação do P_E): 2,5% → 2,7%, dentro do ruído.
- O passo MOPO → DORL-const muda duas coisas: o *survival bonus* e a escala da recompensa no estado (seção 6).

## 2. Janela corrigida para MOPO e DORL-const

**Por quê.** Na semana 9, a correção da janela foi aplicada só a MBPO e DORL. Faltava repetir, sem o recorte, a
decomposição termo a termo MBPO → MOPO → DORL-const → DORL.

**O que foi feito.** MOPO e DORL-const com N = 4c e 8c, 5 sementes cada (`roda_a5.sh`).

**Resultado** (`results/a5/e1_saida.txt`; `docs/semana10.md`, etapa 2). Diferença de Length em NX_0 em cada passo:

| Passo | Termo acrescentado | N = 4 (original) | N = 4c | N = 8c |
|---|---|---|---|---|
| MBPO → MOPO | Penalidade de incerteza | −0,2 (ruído) | +0,2 (ruído) | −0,5 (ruído) |
| MOPO → DORL-const | *Survival bonus* | +8,4 | +5,7 | +3,0 |
| DORL-const → DORL | Variação do P_E | −0,6 (ruído) | +0,8 (ruído) | +0,9 (ruído) |

Conclusões:
- Nas três regras, só o passo MOPO → DORL-const supera o ruído. Ele explica toda a vantagem do DORL sobre o MBPO em
  Length (+6,7 com N = 4c; +3,5 com N = 8c). No Length, as configurações sem bônus empatam entre si com 10 sementes
  (seção 5), então o ganho é do bônus; na cobertura, não (seção 6).
- Cobertura no mesmo passo: 10,4% → 3,3% (N = 4c) e 5,7% → 3,8% (N = 8c). P_U e a variação do P_E não a alteram.
- No FB, MBPO e MOPO ciclam: em 60–62% das trajetórias com N = 4c, a trajetória inteira repete um bloco fixo de
  itens. DORL e DORL-const quase não ciclam (12–14%).
- Ressalva: com N = 8c, as diferenças (+3,0 e +3,5) são de 5 sementes e da ordem do que se mostrou ruído na regra N4
  (seção 5). As sementes 5–9 estão na fila (seção 9).

## 3. Sementes dos bandits

**Por quê.** UCB e ε-greedy tinham uma semente cada, porque cada execução leva várias horas. As conclusões da
semana 8 sobre eles (abaixo do acaso; acima do RL com a sessão fixa) precisavam do dp entre sementes.

**O que foi feito.** Sementes 1–4 de cada um (`roda_bandits.sh`); a execução de 20/10 entra como semente 0.

**Resultado** (NX_0; `results/e1/e1_saida.txt`; `docs/semana10.md`, etapa 3).

| Método | R_tra | R_each | Length |
|---|---|---|---|
| UCB | 3,65 ± 0,04 | 0,869 ± 0,009 | 4,19 ± 0,04 |
| ε-greedy | 3,59 ± 0,12 | 0,841 ± 0,020 | 4,25 ± 0,05 |
| Aleatória | 4,01 ± 0,02 | 0,872 ± 0,001 | 4,60 ± 0,02 |
| RL (MBPO, MOPO, DORL, DORL-const) | 11,7–18,5 | 0,736–0,781 | 15,6–24,0 |

Conclusões:
- Os bandits ficam abaixo da aleatória em R_tra (nos três protocolos) e em Length (NX_0 e FB). Em R_each, o
  ε-greedy fica abaixo nos três protocolos; o UCB empata em NX_0.
- O UCB supera MBPO, MOPO e DORL em R_each (NX_0); contra o DORL-const, fica dentro do dp.
- Em NX_10, o UCB (R_tra 8,45 ± 0,15) supera MBPO e MOPO acima do dp; contra o DORL, a margem é limítrofe; contra o
  DORL-const, fica dentro do ruído. A frase da semana 8 ("em NX_10, os quatro métodos de RL ficam abaixo dos
  bandits") vale só para MBPO e MOPO.
- A afirmação do artigo de que o RL supera os bandits vale para R_tra e Length em NX_0, isto é, para a duração da
  sessão, e não para a recompensa por passo.

## 4. Repetição de visualização

**Por quê.** O artigo não mede quanto a política repete itens. No FB, a repetição é permitida e o item repetido
recebe a mesma recompensa da primeira vez. Sem uma referência real, não se sabe se essa repetição é plausível nem
quanto ela pesa na concentração.

**O que foi feito.** `revisitas.py`: revisitas reais no big_matrix, sem as 968.004 duplicatas exatas (7,7% das
linhas), e repetição nas trajetórias FB de 18 variantes de política, com 5 sementes cada.

**Resultado** (`results/revisitas/`; `docs/semana10.md`, etapa 1).

| Medida | Usuário real | Políticas (FB) |
|---|---|---|
| Repetição dentro da sessão | 3,4% (sessões de 10–30 itens; 70% das revisitas ocorrem em outra sessão) | De 6,7% (MOPO-E6) a 74,3% (MBPO-N4c) |
| Top 10% de popularidade: fração dos repetidos / de todos | 20,3% / 10,6% | Quase iguais (MBPO: 17,3% / 17,8%), abaixo dos 30,6% do acaso |

Conclusões:
- Só a saída calibrada sem bônus (MBPO-E6, MOPO-E6) fica perto da repetição real. Com a regra determinística,
  nenhuma variante fica abaixo de 17%.
- Parte da concentração medida no FB é efeito do protocolo, que permite repetir. No FB, a cobertura sem as repetições
  é de 1,1 a 2,6 vezes a cobertura com elas (MBPO-N1: 6,5% → 16,9%; DORL-E6: 0,9% → 2,3%).
- A repetição real segue a popularidade: as revisitas vão ao top 10% com o dobro da frequência das visualizações. A
  das políticas não segue. Hipótese: no simulador, a repetição é o ciclo que evita a regra de saída, não preferência.
- 20,5% da recompensa de avaliação vem de watch_ratio acima de 1, isto é, do vídeo visto mais de uma vez na mesma
  visualização. A avaliação premia a revisualização dentro da visualização e não mede a revisualização entre passos.

## 5. Varredura do *survival bonus*

**Por quê.** O E2, o E6 e a seção 2 compararam só dois pontos: sem bônus (MOPO) e com o bônus do DORL (8,35).
Faltava saber se o efeito cresce com o tamanho do bônus.

**O que foi feito.** DORL-const com b ∈ {0; 0,5; 1; 2; 4} (`roda_bonus.sh`), nas regras N4 e E6. Na regra N4, 10
sementes por ponto (b = 0,5: 5); na regra E6, 5.

**Resultado, regra N4** (10 sementes; `results/s10/n4_resumo.csv`, `results/s10/alem/`).

| Bônus | Length (NX_0) | Length (FB) | Cobertura |
|---|---|---|---|
| MOPO | 16,69 ± 2,19 | 26,12 ± 2,87 | 9,9 ± 2,7% |
| b = 0 | 17,25 ± 2,25 | 27,39 ± 0,50 | 5,9 ± 0,9% |
| b = 1 | 18,89 ± 2,13 | 25,41 ± 1,13 | 4,8 ± 0,7% |
| b = 2 | 19,34 ± 2,72 | 22,96 ± 1,83 | 5,0 ± 0,8% |
| b = 4 | 19,95 ± 2,06 | 19,64 ± 2,28 | 4,2 ± 0,7% |
| b = 8,35 (DORL-const) | 23,56 ± 2,87 | 14,72 ± 2,53 | 3,3 ± 0,3% |

**Resultado, regra E6** (5 sementes; `results/sb/e6_resumo.csv`). A cobertura vai de 24,3% (b = 0) a 15,3%, 9,9%,
5,5%, 3,0% e 2,5% (b = 8,35). O Length em NX_0 fica entre 21,14 e 21,20 de b = 0 a b = 4 (dp de 0,05 a 0,09).

Conclusões:
- Na regra N4, o Length só sobe com bônus grandes. A referência sem bônus soma b = 0 e o MOPO com a escala do DORL
  (17,41 ± 2,31; 20 execuções). Contra ela, b = 1 e 2 ficam dentro do dp, b = 4 supera por pouco (+2,5) e b = 8,35
  supera com folga (+6,2).
- O FB encurta a cada aumento do bônus, acima do dp em todos os passos e nas duas metades das sementes. É o efeito
  mais robusto do bônus.
- Na regra E6, a cobertura cai com o bônus sem nenhum ganho de Length. Cada passo até b = 4 supera o dp.
- Nenhum bônus leva o R_each acima do acaso (0,872).
- Correção: com 5 sementes, b = 1 e 2 pareciam ganhar +3,7 e +4,3 passos; com 10, esses ganhos não se sustentam.

## 6. Escala da recompensa no estado (`train_max`)

**Por quê.** Com b = 0, a recompensa é a do MOPO, mas a cobertura caía pela metade (E6: 50,2% → 24,3%). Algo além
do bônus muda do MOPO para o DORL-const.

**Mecanismo (verificado no código).** O estado da política inclui a recompensa dividida por `train_max` = MAX_R −
MIN_R (`run_Policy_Main.py:250`; `state_tracker2.py:231–241`). Com λ₂ = 5, o MAX_R ganha o termo de entropia, e o
`train_max` vai de 0,94 (MOPO) para 9,48 em toda a família DORL, mesmo com bônus zero.

**O que foi feito.** Controle nos dois sentidos, trocando só o `train_max` (`tmax.patch`, `roda_tmax.sh`): MOPO com
o `train_max` do DORL (MOPO-tmax) e b = 0 com o do MOPO (SB0-tmin).

**Resultado, regra E6** (5 sementes; `results/alem/alem_resumo.csv`, `results/s10/alem/`, `results/s10/e6_resumo.csv`).

| Configuração | Recompensa | `train_max` | Cobertura | Personalização |
|---|---|---|---|---|
| MOPO-E6 | MOPO | 0,94 | 50,2 ± 5,0% | 0,96 |
| SB0-tmin-E6 | b = 0 | 0,94 | 48,0 ± 8,9% | 0,95 |
| SB0-E6 | b = 0 | 9,48 | 24,3 ± 3,9% | 0,83 |
| MOPO-tmax-E6 | MOPO | 9,48 | 22,8 ± 2,7% | 0,79 |

Na regra N4, com 10 sementes, o MOPO cobre 9,9 ± 2,7%, contra 5,8 ± 0,8% das configurações sem bônus com a escala do
DORL (b = 0 e MOPO-tmax), sem sobreposição entre sementes.

Conclusões:
- Nos dois sentidos, o que decide a cobertura é o `train_max`, não a recompensa.
- Da queda de cobertura do MOPO para o DORL-const, 54% (E6) a 63–65% (N4) já ocorrem com bônus zero. O restante
  acompanha o tamanho do bônus.
- A comparação DORL × MOPO do artigo tem três fatores: a variação do P_E (sem efeito), o *survival bonus* e a escala
  da recompensa no estado, que muda junto com λ₂ sem fazer parte do método.
- Hipótese, não testada: com a faixa maior, o canal de recompensa do estado fica comprimido, o estado distingue menos
  os históricos e a política depende menos dele.

## 7. Avaliação off-policy no log real

**Por quê.** Todos os resultados anteriores dependem do simulador e da regra de saída. A avaliação off-policy (OPE)
estima o valor de uma política com dados gerados por outra, sem simulador.

**O que foi feito** (`docs/off_policy.md`).
- **Log real:** 9.454.659 passos (623.916 sessões) de 5.765 usuários fora da avaliação, com os itens do small_matrix.
  Os usuários de avaliação não têm interação com esses itens no big_matrix.
- **Estimador:** SNIPS, com a política de registro estimada por contagens (último item → item). Foi escolhido em duas
  validações semi-sintéticas: com a política de registro estimada, o IPS fica enviesado e o SNIPS não.
- **Políticas:** as 20 do simulador (4 métodos × 5 sementes), na época 200, retreinadas com checkpoint (`roda_ckpt.sh`)
  e com logs idênticos aos originais.
- **Alcance:** o valor é por passo, e a política é aplicada sem a máscara de itens já vistos.

**Resultado** (`results/ope/ope_saida.txt`, `results/ope/ope_comparacao.txt`). Política de registro: 0,881;
política uniforme: 0,898.

| Método | SNIPS (média ± dp) | Mediana | ESS | Itens com 95% da exposição |
|---|---|---|---|---|
| MBPO | 0,367 ± 0,309 | 0,278 | 1.225–2.911 | 2–4 |
| MOPO | 0,350 ± 0,271 | 0,331 | 215–2.806 | 3–4 |
| DORL | 0,639 ± 0,528 | 0,389 | 1.106–1.554 | 2–4 |
| DORL-const | 0,630 ± 0,633 | 0,384 | 1.507–2.427 | 4–16 |

Conclusões:
- As políticas são quase determinísticas nos contextos reais: 95% da exposição cai em 2 a 16 dos 3.327 itens. O Gini
  da exposição vai de 0,997 a 1,000, contra 0,244 da política de registro.
- 17 das 20 ficam abaixo da política de registro (IC95 inteiro abaixo em 15; inteiro acima em 2).
- A ordem dos métodos não é identificável: a maior diferença entre médias (0,29) não supera o dp entre sementes
  (0,27–0,63).
- O suporte é pequeno: ESS de 215 a 2.911 passos (0,002% a 0,03% do log). O IC95 cobre só a amostragem de usuários.

## 8. Protocolo de avaliação e síntese

**Protocolo** (`docs/protocolo_avaliacao.md`, versão 0). Define seis objetivos de um RLRS, cada um com pergunta,
métrica mínima, referências obrigatórias e controles:
- O1 satisfação imediata (R_each);
- O2 valor de longo prazo (Length e R_tra, sempre com R_each e NX_10);
- O3 diversidade e concentração (categoria, item, usuário e sessão);
- O4 popularidade (efeito Mateus);
- O5 fidelidade do sinal de treino;
- O6 plausibilidade do ambiente.

Os sete controles são: (a) desvio entre sementes; (b) referência sem sinal; (c) ablação por constante; (d) variação
da regra e do protocolo; (e) concentração em vários níveis; (f) referência empírica; (g) avaliação off-policy.

**Síntese** (`docs/sintese.md`, `docs/tabelas/sintese.tex`). Registra cada afirmação do artigo (A1–A7) e cada achado
deste trabalho (B1–B8) com o sinal examinado, o controle e a força da evidência. Força das afirmações do artigo:

| # | Afirmação do artigo | Força |
|---|---|---|
| A1 | O DORL prolonga a sessão | Condicional à regra determinística |
| A2 | C1: a penalidade de incerteza produz efeito Mateus | Dentro do ruído |
| A3 | C4: o termo de entropia reintroduz exploração | Não sustentada |
| A4 | O DORL reduz a concentração | Invertida no nível do item |
| A5 | O RL supera os bandits | Condicional |
| A6 | A Tabela 2 representa cada método | Não sustentada com uma semente |
| A7 | C2 e C3 | Não testada |

## 9. Pendente

Sementes 5–9 com N = 8c (MBPO, DORL, MOPO e DORL-const), na fila da 4090 desde 17/11. Elas servem para confirmar, em
Length, DORL − MBPO (+3,5) e MOPO → DORL-const (+3,0), hoje com 5 sementes. Diferenças dessa ordem com 5 sementes já
se mostraram ruído (seção 5). O resultado entra no relatório.
