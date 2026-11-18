# Semana 10 — últimos experimentos e protocolo de avaliação

Registro iniciado em 11/11/2026. Mesma configuração e mesmos critérios da semana 9 (`docs/semana9.md`): média ± dp
entre 5 sementes; uma diferença conta quando supera o maior dp entre sementes dos dois métodos.

| Id | Experimento | Status |
|---|---|---|
| Rev | Revisualização de vídeos: referência real e repetição das políticas no FB (`revisitas.py`) | Feito (A, B, C) |
| A5-2 | Janela corrigida (N = 4, 8) para MOPO e DORL-const (`roda_a5.sh`) | Feito |
| Band | Sementes 1–4 de UCB e ε-greedy (`roda_bandits.sh`) | Feito |
| SB | Varredura do *survival bonus*: b ∈ {0; 0,5; 1; 2; 4}, regras N4 e E6 (`roda_bonus.sh`) | Feito |
| TM | Controle da varredura: MOPO com o `train_max` do DORL (`tmax.patch`, `roda_tmax.sh`) | Feito (16/11) |
| S10 | Sementes 5–9 na regra N4 (MOPO, MOPO-tmax, b = 0, 1, 2, 4, DORL-const) e controle inverso no E6 | Feito (17/11) |
| S10-8c | Sementes 5–9 com N = 8c (MBPO, MOPO, DORL-const, DORL; `roda_a5.sh 8`) | Feito (18/11; etapa 5) |
| Prot | Protocolo de avaliação de RLRS (`docs/protocolo_avaliacao.md`) | Versão 0 |
| Sínt | Tabela de síntese (`docs/sintese.md`, `docs/tabelas/sintese.tex`) | Feito; B7 acrescentada |

## Etapa 1 — Revisualização de vídeos

A repetição de itens é a métrica que o DORL não usa e que mais muda a leitura da concentração. As políticas repetem
de 2 a 20 vezes mais que o usuário real e, nelas, a repetição não segue a popularidade, ao contrário do usuário real.

**Pergunta.** Como o usuário real revê vídeos, e como isso se compara à repetição das políticas no FB? A repetição
altera a concentração medida?

**Justificativa.** A repetição entra na avaliação do DORL em dois lugares sem ser nomeada:

- **Loop dentro da visualização:** a recompensa de avaliação é o watch_ratio bruto limitado a 5 (`KuaiEnv.load_mat`),
  e watch_ratio > 1 é o vídeo visto mais de uma vez na mesma visualização.
- **Repetição entre passos:** no FB a política pode repetir itens, e o item repetido recebe a mesma recompensa da
  primeira vez (`KuaiEnv.py:317`). Nenhuma métrica do artigo mede quanto a política repete.

Sem uma referência real, não se sabe se a repetição do FB é plausível nem quanto ela pesa na concentração.

**Procedimento.** `revisitas.py`, em três partes.

- **A. Revisitas no big_matrix:**
  - remove duplicatas exatas (mesmo usuário, vídeo, timestamp e watch_ratio);
  - classifica cada nova ocorrência de um par usuário–vídeo pelo intervalo desde a anterior: imediata (≤ 60 s),
    mesma sessão (60 s a 30 min) ou outra sessão (> 30 min);
  - mede a repetição dentro das sessões (corte de 30 min) com a mesma definição do `fb_rotacao.py`:
    1 − itens distintos ÷ tamanho.
- **B. Loop no small_matrix:** fração das células com watch_ratio > 1 e parte da recompensa de avaliação que vem
  de watch_ratio acima de 1.
- **C. Trajetórias FB das 18 variantes** (5 sementes cada), com quatro medidas:
  - repetição por trajetória;
  - fração dos passos e dos passos repetidos em itens do top 10% de popularidade (big_matrix);
  - cobertura da época contando as repetições (distintos ÷ passos);
  - cobertura sem as repetições (distintos ÷ soma dos distintos por trajetória).

A e B rodaram no Mac; C, na 4090.

**Resultado A e B** (`results/revisitas/revisitas_saida.txt`).

| Medida | Usuários de avaliação | Todos |
|---|---|---|
| Duplicatas exatas removidas | — | 968.004 (7,7% do big_matrix) |
| Visualizações que repetem um par | 5,5% | 10,9% |
| Repetições: imediata / mesma sessão / outra sessão | 28,1% / 1,9% / 69,9% | 23,7% / 2,6% / 73,6% |
| Repetição dentro da sessão (sessões de 10–30 itens) | 3,4% | 2,7% |
| Top 10% de popularidade: revisitas / visualizações | 20,3% / 10,6% | 65,6% / 40,2% |
| Gini por item: revisitas / visualizações | 0,958 / 0,753 | 0,877 / 0,670 |
| watch_ratio da revisita ÷ visualização anterior (mediana) | 1,00 | 1,00 |

Small_matrix: 32,4% das células têm watch_ratio > 1, e 20,5% da recompensa de avaliação vem da parte acima de 1.

Entre os usuários de avaliação, 11,8% das linhas repetem um par; 56% dessas repetições são duplicatas exatas. Sem as
duplicatas, sobram 5,5% de repetições, 70% delas em outra sessão. Uma versão anterior das notas citava 62% de duplicatas, com outro
denominador. O número deste script é o que vale.

**Resultado C** (FB; média ± dp entre 5 sementes; `results/revisitas/revisitas_resumo.csv`). Taxa de base da
política uniforme no top 10%: 30,6% dos 3.327 itens do small_matrix. O número foi calculado à parte e entra no
`revisitas.py` a partir da reexecução (`results/revisitas/revisitas_saida.txt`).

| Método | Repetição | Passos no top 10% | Repetidos no top 10% | Cobertura com repetição | Cobertura sem repetição |
|---|---|---|---|---|---|
| MBPO | 0,718 ± 0,040 | 0,178 ± 0,071 | 0,173 ± 0,080 | 0,039 ± 0,007 | 0,066 ± 0,020 |
| MOPO | 0,700 ± 0,087 | 0,183 ± 0,171 | 0,180 ± 0,177 | 0,034 ± 0,005 | 0,055 ± 0,009 |
| DORL | 0,380 ± 0,064 | 0,132 ± 0,108 | 0,114 ± 0,130 | 0,024 ± 0,003 | 0,036 ± 0,003 |
| DORL-const | 0,315 ± 0,058 | 0,305 ± 0,072 | 0,357 ± 0,087 | 0,028 ± 0,005 | 0,039 ± 0,005 |
| MBPO-E5 | 0,357 ± 0,034 | 0,303 ± 0,057 | 0,035 ± 0,077 | 0,013 ± 0,002 | 0,018 ± 0,002 |
| DORL-E5 | 0,354 ± 0,064 | 0,228 ± 0,115 | 0,239 ± 0,160 | 0,025 ± 0,005 | 0,036 ± 0,004 |
| MBPO-E6 | 0,081 ± 0,047 | 0,255 ± 0,026 | 0,130 ± 0,064 | 0,438 ± 0,065 | 0,477 ± 0,047 |
| MOPO-E6 | 0,067 ± 0,040 | 0,259 ± 0,024 | 0,162 ± 0,066 | 0,410 ± 0,045 | 0,442 ± 0,030 |
| DORL-E6 | 0,615 ± 0,011 | 0,112 ± 0,114 | 0,110 ± 0,121 | 0,009 ± 0,000 | 0,023 ± 0,001 |
| DORL-const-E6 | 0,615 ± 0,025 | 0,134 ± 0,118 | 0,130 ± 0,119 | 0,009 ± 0,001 | 0,024 ± 0,001 |
| MBPO-N1 | 0,694 ± 0,091 | 0,302 ± 0,216 | 0,308 ± 0,257 | 0,065 ± 0,010 | 0,169 ± 0,020 |
| DORL-N1 | 0,558 ± 0,117 | 0,228 ± 0,151 | 0,240 ± 0,236 | 0,012 ± 0,003 | 0,025 ± 0,003 |
| MBPO-N4c | 0,743 ± 0,015 | 0,168 ± 0,154 | 0,165 ± 0,162 | 0,032 ± 0,003 | 0,054 ± 0,008 |
| DORL-N4c | 0,371 ± 0,078 | 0,207 ± 0,084 | 0,234 ± 0,129 | 0,024 ± 0,004 | 0,035 ± 0,005 |
| MBPO-N8 | 0,552 ± 0,033 | 0,152 ± 0,052 | 0,151 ± 0,054 | 0,042 ± 0,005 | 0,052 ± 0,006 |
| DORL-N8 | 0,266 ± 0,036 | 0,221 ± 0,106 | 0,236 ± 0,115 | 0,034 ± 0,002 | 0,043 ± 0,003 |
| MBPO-N8c | 0,480 ± 0,079 | 0,215 ± 0,130 | 0,198 ± 0,159 | 0,046 ± 0,011 | 0,056 ± 0,011 |
| DORL-N8c | 0,178 ± 0,040 | 0,189 ± 0,086 | 0,205 ± 0,122 | 0,037 ± 0,004 | 0,044 ± 0,003 |

**Interpretação.**

1. **As políticas repetem de 2 a 20 vezes mais que o usuário real.**
   - Nas trajetórias FB, a repetição vai de 6,7% (MOPO-E6) a 74,3% (MBPO-N4c); nas sessões reais de 10–30 itens,
     é de 3,4%.
   - Só a saída calibrada sem *survival bonus* (MBPO-E6, MOPO-E6) chega perto. Com a regra determinística, nenhuma
     variante fica abaixo de 17%.
   - O usuário real quase não repete dentro da sessão: 70% das repetições acontecem em outra sessão.
2. **A repetição infla a concentração.**
   - A cobertura com repetições é sempre menor que a cobertura sem elas, de 1,1 a 2,6 vezes. Exemplos: MBPO-N1
     16,9% → 6,5%; DORL-E6 2,3% → 0,9%.
   - Parte da concentração medida no FB é, portanto, efeito do protocolo, que permite repetir, e não da escolha
     de itens.
3. **A repetição das políticas não segue a popularidade; a do usuário real segue.**
   - Usuário real: as revisitas vão ao top 10% com o dobro da frequência das visualizações (20,3% contra 10,6%),
     e o Gini por item sobe de 0,753 para 0,958.
   - Políticas: os passos repetidos caem no top 10% com a mesma frequência que os passos em geral (MBPO 17,3%
     contra 17,8%), quase sempre abaixo dos 30,6% do acaso.
   - Hipótese compatível com a etapa 7.1 da semana 9: no simulador, a repetição é o ciclo que evita a regra de
     saída, não preferência.
   - Exceções: MBPO-E5 (3,5% contra 30,3%) e MBPO-E6 (13,0% contra 25,5%) repetem itens menos populares que os
     que recomendam.
4. **A concentração do RL no simulador não é de popularidade.** O RL concentra-se em poucos itens quaisquer, em geral
   menos populares que o acaso. O "efeito Mateus" medido aqui é homogeneização, não reforço do que já é popular
   (etapa 8 da semana 9). O mecanismo de reforço que a revisita real tem (Gini 0,958) não existe no simulador.
5. **Diferença sem interpretação firme (descartada na etapa 2).** DORL 13,2% contra DORL-const 30,5% de passos no top 10% (dp 0,108).
   - A diferença é compatível com o termo de k = 1 favorecer itens raros (S1: Spearman −0,53).
   - Mas some sob o E6 (11,2% contra 13,4%). Leitura: possível efeito da variação do P_E na popularidade, só na
     regra determinística.
6. **A recompensa de avaliação já premia o loop.** 20,5% dela vem de watch_ratio acima de 1. O sinal que a avaliação
   mede inclui a revisualização dentro da visualização, enquanto a revisualização entre passos não é medida.

**Limitações.**

- A referência de popularidade dos usuários de avaliação (10,6% das visualizações no top 10%) não é comparável à
  das políticas. Verificado em 14/11: o big_matrix desses usuários não tem nenhuma linha com os 3.327 itens do small_matrix
  (0 de 533.090); os demais usuários têm 9,45 milhões de linhas com eles.
- A repetição das políticas é medida por trajetória simulada, e a do usuário por sessão real com corte de 30 min.
- O dp da popularidade é grande (até 0,26).

## Etapa 2 — Janela corrigida para MOPO e DORL-const

A decomposição do E2 se repete com a janela corrigida, nos dois N: só o *survival bonus* (o bônus fixo por passo que o termo de entropia soma) muda o resultado.
A penalidade de incerteza e a variação do P_E ficam dentro do ruído em Length, cobertura e ciclos do FB.

**Pergunta.** A decomposição MBPO → MOPO → DORL-const → DORL (E2, E6) vale com a regra de saída sem o recorte da
janela (A5)?

**Procedimento.** MOPO e DORL-const com N = 4 e N = 8 corrigidos, 5 sementes cada (`METODOS="MOPO DORLconst"
roda_a5.sh`; DORL-const com `PE_CONST=1.9609`). As demais análises usam as mesmas trajetórias:
`e1_protocolos.py` → `results/a5/e1_saida.txt`; `alem_da_acuracia.py`, `bolha.py`, `fb_rotacao.py` e
`revisitas.py` → `results/{alem,bolha,fb,revisitas}/`.

**Resultado.** Média entre 5 sementes. Length e R_tra em NX_0; cobertura e personalização em NX_10; ciclo puro no FB.

| N | Método | Length | R_tra | Cobertura | Personalização | Ciclo puro (FB) |
|---|---|---|---|---|---|---|
| 4 (original) | MBPO | 15,78 ± 1,83 | 11,69 ± 2,32 | 12,0% | 0,52 | 0,53 |
| | MOPO | 15,60 ± 2,34 | 12,00 ± 2,61 | 11,2% | 0,52 | 0,60 |
| | DORL-const | 24,00 ± 2,66 | 18,53 ± 0,29 | 3,4% | 0,63 | 0,05 |
| | DORL | 23,42 ± 1,10 | 18,27 ± 2,46 | 3,3% | 0,59 | 0,14 |
| 4c | MBPO | 16,65 ± 1,95 | 12,20 ± 2,03 | 10,5% | 0,48 | 0,62 |
| | MOPO | 16,84 ± 1,26 | 12,89 ± 0,75 | 10,4% | 0,45 | 0,60 |
| | DORL-const | 22,57 ± 2,46 | 18,65 ± 3,63 | 3,3% | 0,56 | 0,12 |
| | DORL | 23,39 ± 2,61 | 18,62 ± 3,10 | 3,2% | 0,54 | 0,14 |
| 8c | MBPO | 18,64 ± 1,72 | 13,31 ± 2,46 | 5,3% | 0,21 | 0,14 |
| | MOPO | 18,12 ± 1,44 | 12,36 ± 1,31 | 5,7% | 0,22 | 0,36 |
| | DORL-const | 21,15 ± 1,17 | 15,74 ± 2,01 | 3,8% | 0,60 | 0,08 |
| | DORL | 22,10 ± 0,83 | 17,39 ± 0,68 | 3,7% | 0,55 | 0,07 |

Decomposição em Length (NX_0); "acima" = supera o maior dp entre sementes dos dois métodos.

| Passo | Termo acrescentado | N = 4 | N = 4c | N = 8c |
|---|---|---|---|---|
| MBPO → MOPO | P_U | −0,2 (ruído) | +0,2 (ruído) | −0,5 (ruído) |
| MOPO → DORL-const | *Survival bonus* | +8,4 (acima) | +5,7 (acima) | +3,0 (acima) |
| DORL-const → DORL | Variação do P_E | −0,6 (ruído) | +0,8 (ruído) | +0,9 (ruído) |

**Revisto com 10 sementes (18/11; etapa 5):** com N = 8c, os três passos ficam dentro do ruído em Length (−0,8; +1,8;
+1,2), e DORL − MBPO também (+2,2, dp 2,3).

**Interpretação.**

1. **O E2 se reproduz com a janela corrigida.** Nos três cenários, o único passo acima do ruído é o *survival bonus*.
   Ele explica toda a vantagem do DORL sobre o MBPO em Length (+6,7 com N = 4c; +3,5 com N = 8c). Em R_tra, com
   N = 8c, DORL − DORL-const = +1,6, dentro do dp (2,0).
   - Revisto com 10 sementes (18/11; etapa 5): com N = 8c, nenhum passo supera o ruído em Length, e não há vantagem
     do DORL a explicar (+2,2, dp 2,3). Em R_tra, DORL − DORL-const = +2,5, pouco acima do dp (2,2), e
     MOPO → DORL-const = +2,2, no limite do dp (2,2).
2. **A concentração de itens também vem do bônus.**
   - A cobertura cai de 10,4% para 3,3% (N = 4c) e de 5,7% para 3,8% (N = 8c) no passo MOPO → DORL-const
     (5 sementes; com N = 8c, as sementes 5–9 não têm cobertura calculada, ver etapa 5).
   - P_U e a variação do P_E não a alteram.
   - É o mesmo padrão do E6 (50,2% → 2,5%), agora em três regras determinísticas e na calibrada.
3. **O ciclo do FB é dos métodos sem bônus.**
   - MBPO e MOPO ciclam: ciclo puro em 0,60–0,62 das trajetórias com N = 4c.
   - DORL e DORL-const quase não ciclam (0,12–0,14).
   - Responde em parte à questão em aberto da semana 9 ("por que DORL/DORL-const quase não ciclam no FB"): o que
     separa os dois grupos é o bônus, não a variação do P_E. O mecanismo segue em aberto.
4. **Bolha:** nenhuma variante forma bolha individual. V vai de 9,70 a 10,05, contra 8,02 do usuário.
5. **A diferença de popularidade da etapa 1 (item 5) não se sustenta.** DORL-const contra DORL no top 10%:
   - N = 4: 30,5% contra 13,2%;
   - N = 4c: 21,5% contra 20,7%;
   - N = 8c: 30,9% contra 18,9%.

   A direção não é estável entre as regras, e os dp são de 0,08 a 0,12. Leitura: ruído, não efeito do P_E.
6. **Personalização com N = 8c.** MBPO e MOPO caem para 0,21–0,22, abaixo de DORL e DORL-const (0,55–0,60).
   Fato registrado, sem interpretação.

**Limitações.** Uma constante do bônus (a do E2) e um λ₁. A aleatória com janela corrigida está em
`results/e3/aleatoria_N{4,8}c.json` e não entra no `e1_saida.txt` do A5; a de N = 8c entra no resumo de 10 sementes
(`results/s10/n8c_saida.txt`, etapa 5).

## Etapa 3 — Sementes dos bandits

Com 5 sementes, a S3 se mantém: os bandits ficam abaixo da política aleatória em R_tra e Length nos três protocolos.
O RL fica abaixo dos bandits em R_each, e em R_tra quando o comprimento é fixado.

**Pergunta.** As conclusões da S3 e do E1 sobre os bandits, tiradas com uma semente, se mantêm com cinco?

**Procedimento.** UCB e ε-greedy com as sementes 1 a 4 (`roda_bandits.sh`; mensagens `UCB_s{n}` e
`epsilon-greedy_s{n}`), mesmos argumentos da execução de 20/10, que entra como semente 0. Agregação pelo
`e1_protocolos.py` → `results/e1/e1_saida.txt`. Cada execução levou cerca de 15 h, com quatro em paralelo.

**Resultado** (média ± dp entre 5 sementes).

| Protocolo | Método | R_tra | R_each | Length | MCD |
|---|---|---|---|---|---|
| NX_0 | UCB | 3,65 ± 0,04 | 0,869 ± 0,009 | 4,19 ± 0,04 | 0,822 |
| | ε-greedy | 3,59 ± 0,12 | 0,841 ± 0,020 | 4,25 ± 0,05 | 0,818 |
| | Aleatória | 4,01 ± 0,02 | 0,872 ± 0,001 | 4,60 ± 0,02 | 0,827 |
| NX_10 | UCB | 8,45 ± 0,15 | 0,845 ± 0,014 | 10 | 0,814 |
| | ε-greedy | 8,29 ± 0,16 | 0,829 ± 0,016 | 10 | 0,811 |
| | Aleatória | 8,72 ± 0,03 | 0,872 ± 0,003 | 10 | 0,827 |
| FB | UCB | 2,95 ± 0,03 | 0,919 ± 0,004 | 3,21 ± 0,03 | 0,838 |
| | ε-greedy | 2,12 ± 0,05 | 0,789 ± 0,021 | 2,68 ± 0,01 | 0,815 |
| | Aleatória | 4,01 ± 0,01 | 0,873 ± 0,004 | 4,59 ± 0,01 | 0,827 |

**Interpretação.**

1. **Os bandits continuam abaixo do acaso** em R_tra e Length nos três protocolos, acima do dp. Em R_each:
   - o ε-greedy fica abaixo nos três protocolos;
   - o UCB empata com o acaso em NX_0 (0,869 contra 0,872, dentro do dp) e fica abaixo em NX_10;
   - a única exceção é o FB, em que o UCB supera o acaso (0,919 contra 0,873), provavelmente porque repete itens
     de recompensa alta. Hipótese não verificada: os bandits não têm trajetórias registradas.

   A S3 dizia "abaixo do acaso em R_tra, R_each e Length". Com 5 sementes, a frase vale para o ε-greedy; para o
   UCB, vale em R_tra e Length, não em R_each.
2. **RL × bandits** (contra o maior dp entre sementes):
   - **NX_0, R_tra e Length:** o RL supera os bandits por larga margem, como na Tabela 2.
   - **R_each:** o UCB supera MBPO, MOPO e DORL em NX_0; contra o DORL-const, fica dentro do dp (0,088 contra 0,095).
   - **NX_10, R_tra:** o UCB supera MBPO e MOPO acima do dp. Contra o DORL, a margem é limítrofe (1,260 contra
     1,258); contra o DORL-const, fica dentro do dp. O ε-greedy supera MBPO e MOPO; contra DORL e DORL-const,
     fica dentro do dp.

   A frase do E1, "em NX_10 os quatro métodos de RL ficam abaixo dos bandits", só se sustenta para MBPO e MOPO;
   para DORL e DORL-const é ruído.
3. **A afirmação A5 do artigo (o RL supera os bandits) continua condicional.** Vale para R_tra e Length com a regra
   de saída determinística, e se inverte em R_each e, para MBPO e MOPO, em NX_10.
4. **A Tabela 2 nos bandits.** UCB: 3,61 / 0,853 / 4,22 / 0,811 no artigo, contra 3,65 / 0,869 / 4,19 / 0,822 aqui.
   ε-greedy: 3,52 / 0,828 / 4,22 / 0,823, contra 3,59 / 0,841 / 4,25 / 0,818. Os valores ficam próximos; não há
   dp entre sementes no artigo para comparar.

**Limitações.** Sem trajetórias registradas para os bandits (o `traj_log.patch` cobre só o avaliador das
políticas de RL), então cobertura, bolha e repetição não foram medidas para eles.

## Etapa 4 — Varredura do *survival bonus*

**Pergunta.** O efeito do *survival bonus* (bônus fixo por passo) cresce com o seu tamanho? O E2, o E6 e a etapa 2
compararam só dois pontos: MOPO (sem bônus) e DORL-const (b = 8,35).

**Desenho.** DORL-const com outras constantes (`roda_bonus.sh`, 14/11): b = λ₂·(c − 0,291) ∈ {0; 0,5; 1; 2; 4},
λ₁ = 0,05, λ₂ = 5, 5 sementes, nas regras N4 (determinística original) e E6 (saída calibrada, N = 1). São 50
execuções `SB{b}_{N4,E6}_s*`, todas com 200 épocas (`results/sb/n4_saida.txt`, `e6_saida.txt`). Os extremos já
existiam: MOPO e DORL-const (b = 8,35).

**O que b = 0 controla (verificado no código).** Com b = 0, a recompensa é a do MOPO: o MIN_R não depende do
`PE_CONST` (`simulated_env.py:81`). O valor exato do mínimo é 0,29089 (`results/analises/pe_diag.json`), então o b
real é o nominal + 0,0005. O que muda do MOPO para b = 0 é a escala da recompensa no estado de treino:
`train_max = MAX_R − MIN_R` (`run_Policy_Main.py:250`, usado só em `state_tracker2.py:231`), e o MAX_R inclui
λ₂·entropy_max (`simulated_env.py:82`). Com λ₂ = 5, a faixa cresce 5·(2 − 0,291) = 8,55, igual em toda a série SB e
no DORL-const. A série SB isola, portanto, o tamanho do bônus; o passo MOPO → b = 0 isola a normalização do estado.

**Resultado, regra N4** (cobertura e personalização em NX_10; `results/sb/n4_resumo.csv`, `results/alem/`, `results/fb/`).

| Bônus | Length (NX_0) | R_tra (NX_0) | R_each (NX_0) | Cobertura (%) | Length (FB) | Repetição (FB) | Ciclo puro (FB) |
|---|---|---|---|---|---|---|---|
| MOPO (λ₂ = 0) | 15,60 ± 2,34 | 12,00 ± 2,61 | 0,763 ± 0,061 | 11,2 ± 3,2 | 26,13 ± 3,63 | 0,70 ± 0,09 | 0,60 ± 0,24 |
| b = 0 | 15,86 ± 2,44 | 11,26 ± 1,95 | 0,712 ± 0,090 | 6,1 ± 1,1 | 27,68 ± 0,24 | 0,73 ± 0,01 | 0,70 ± 0,19 |
| b = 0,5 | 17,41 ± 1,16 | 12,97 ± 0,65 | 0,749 ± 0,069 | 5,7 ± 1,0 | 25,93 ± 0,92 | 0,66 ± 0,04 | 0,40 ± 0,31 |
| b = 1 | 19,59 ± 2,67 | 14,38 ± 2,40 | 0,736 ± 0,079 | 4,6 ± 0,8 | 25,85 ± 1,00 | 0,64 ± 0,03 | 0,31 ± 0,28 |
| b = 2 | 20,19 ± 1,73 | 14,29 ± 1,87 | 0,706 ± 0,054 | 5,0 ± 1,1 | 23,57 ± 2,03 | 0,56 ± 0,08 | 0,15 ± 0,24 |
| b = 4 | 19,83 ± 2,21 | 15,97 ± 1,56 | 0,812 ± 0,111 | 3,8 ± 0,3 | 21,56 ± 0,85 | 0,53 ± 0,02 | 0,15 ± 0,17 |
| b = 8,35 (DORL-const) | 24,00 ± 2,66 | 18,53 ± 0,29 | 0,781 ± 0,095 | 3,4 ± 0,4 | 14,31 ± 2,23 | 0,31 ± 0,06 | 0,05 ± 0,01 |
| Aleatória | 4,60 ± 0,02 | 4,01 ± 0,02 | 0,872 ± 0,001 | 86,5 | 4,59 ± 0,01 | — | — |

**Resultado, regra E6** (`results/sb/e6_resumo.csv`).

| Bônus | Length (NX_0) | R_each (NX_0) | MCD (NX_0) | Cobertura (%) | Personalização | Repetição (FB) |
|---|---|---|---|---|---|---|
| MOPO-E6 (λ₂ = 0) | 20,75 ± 0,09 | 0,842 ± 0,008 | 0,669 ± 0,019 | 50,2 ± 5,0 | 0,96 ± 0,02 | 0,07 ± 0,04 |
| b = 0 | 21,14 ± 0,09 | 0,839 ± 0,030 | 0,563 ± 0,019 | 24,3 ± 3,9 | 0,83 ± 0,03 | 0,23 ± 0,04 |
| b = 0,5 | 21,17 ± 0,06 | 0,813 ± 0,034 | 0,594 ± 0,026 | 15,3 ± 1,1 | 0,79 ± 0,01 | 0,27 ± 0,01 |
| b = 1 | 21,16 ± 0,07 | 0,849 ± 0,030 | 0,590 ± 0,026 | 9,9 ± 0,6 | 0,73 ± 0,02 | 0,32 ± 0,01 |
| b = 2 | 21,20 ± 0,07 | 0,855 ± 0,045 | 0,618 ± 0,029 | 5,5 ± 0,2 | 0,60 ± 0,02 | 0,42 ± 0,01 |
| b = 4 | 21,19 ± 0,05 | 0,861 ± 0,063 | 0,667 ± 0,060 | 3,0 ± 0,2 | 0,39 ± 0,03 | 0,51 ± 0,01 |
| b = 8,35 (DORL-const-E6) | 20,91 ± 0,23 | 0,864 ± 0,032 | 0,806 ± 0,059 | 2,5 ± 0,5 | 0,16 ± 0,02 | 0,62 ± 0,03 |
| Aleatória | 20,10 ± 0,05 | 0,872 ± 0,002 | 0,827 ± 0,001 | 86,5 | 1,00 | — |

**Interpretação.**

1. **O Length responde à dose na regra N4, sem ser linear.** b = 0 reproduz o MOPO (15,9 contra 15,6). De b = 0 para
   b = 1 e b = 2, o ganho é de +3,7 e +4,3 passos, acima do dp (2,7 e 2,4). Entre b = 1, 2 e 4 não há diferença
   (19,6 a 20,2). De b = 4 para 8,35, mais +4,2, acima do dp (2,7). O ganho de Length do MOPO para o DORL-const vem
   todo do bônus: a conclusão do E2 se mantém.
   - **Revisto com 10 sementes (17/11; seção "Sementes 5–9" adiante).** Com 5 sementes, MOPO-tmax e b = 0, que
     são a mesma configuração, diferiam em 2,9 passos. Com 10, empatam, e a diferença inverte o sinal nas sementes
     5–9. Contra a referência de bônus zero (17,41 ± 2,31, 20 execuções), b = 1 e b = 2 ficam dentro do dp; b = 4
     supera por margem pequena (+2,5, dp 2,3); b = 8,35 supera com folga (+6,2). Os ganhos de +3,7 e +4,3 citados
     acima, de 5 sementes, não se sustentam.
   - A conclusão do E2 (DORL-const = DORL) não muda. Do MOPO para o DORL-const, o ganho de Length é do bônus: as três
     configurações sem bônus (MOPO, MOPO-tmax e b = 0) não diferem entre si.
2. **Na regra E6, nenhuma dose compra Length.** De b = 0 a b = 4, o Length fica entre 21,14 e 21,20, variação menor
   que o dp (0,05 a 0,09). Confirma o E6 com cinco pontos em vez de dois.
3. **A cobertura cai com a dose, de forma monótona, na regra E6:** 24,3% → 15,3% → 9,9% → 5,5% → 3,0% → 2,5%. Cada
   passo até b = 4 supera o dp; o último (3,0% → 2,5%) não. A personalização cai junto (0,83 → 0,16), e a repetição
   no FB sobe (0,23 → 0,62). Na regra N4, a queda é menor e não é monótona ponto a ponto (6,1% → 5,7% → 4,6% → 5,0% → 3,8% → 3,4%); de
   b = 0 para b = 4, a diferença (−2,3 pontos) supera o dp (1,1). Com 10 sementes na regra N4, o padrão se repete
   (5,9% → 4,8% → 5,0% → 4,2% → 3,3% para b = 0, 1, 2, 4 e 8,35).
4. **Correção: b = 0 não reproduz o MOPO em cobertura.** Com a mesma recompensa, a cobertura cai de 50,2% para 24,3%
   no E6 e de 11,2% para 6,1% no N4, sem sobreposição entre sementes (E6: 43,6–55,6% contra 19,9–28,5%; N4:
   8,6–16,3% contra 4,3–7,2%). No E6, a MCD também muda (0,669 → 0,563) e o Length sobe 0,39 (dp 0,09).
   - Consequência: a frase "do MOPO para o DORL-const muda só o bônus fixo por passo" (E2, E6 e etapa 2) está
     errada. Mudam duas coisas: o bônus e a normalização da recompensa no estado de treino.
   - Da queda de cobertura do MOPO para o DORL-const, a parte que ocorre com bônus zero é 25,9 de 47,7 pontos no
     E6 (54%) e 5,1 de 7,8 pontos no N4 (65%; com 10 sementes, 4,2 de 6,7 pontos, 63%). O restante acompanha o
     tamanho do bônus.
   - Causa, testada em 16/11 (`MOPO-tmax`: MOPO com `TMAX_LAMBDA=5`; `results/alem/alem_resumo.csv`): é a escala do
     canal de recompensa do estado de treino. O estado concatena a recompensa dividida por `train_max` e limitada a
     1 (`state_tracker2.py:231–241`); o patch muda o `train_max` de 0,9387 para 9,4842 (linha `TMAX_LAMBDA` dos
     logs) e não toca na recompensa. Só com isso, a cobertura do MOPO cai para a de b = 0:

     | Cobertura (NX_10) | MOPO | MOPO-tmax | b = 0 | DORL-const |
     |---|---|---|---|---|
     | Regra N4 | 11,2 ± 3,2% | 5,4 ± 0,9% | 6,1 ± 1,1% | 3,4 ± 0,4% |
     | Regra E6 | 50,2 ± 5,0% | 22,8 ± 2,7% | 24,3 ± 3,9% | 2,5 ± 0,5% |

     Não há sobreposição entre sementes do MOPO e do MOPO-tmax (E6: 43,6–55,6% contra 20,5–27,1%; N4: 8,6–16,3%
     contra 4,3–6,5%), e a diferença entre MOPO-tmax e b = 0 fica dentro do dp (1,5 e 0,7 ponto). A personalização
     acompanha (E6: 0,958 → 0,794, contra 0,831 de b = 0; N4: 0,524 → 0,313, contra 0,384).
     Confirmado no sentido inverso em 17/11: b = 0 com o `train_max` do MOPO volta à cobertura do MOPO no E6
     (48,0% contra 50,2%; seção "Sementes 5–9" adiante).
   - Leitura: a comparação DORL × MOPO do artigo tem três fatores, não dois: a variação do P_E (sem efeito, E2), o
     *survival bonus* e a escala da recompensa no estado, que muda junto com λ₂ sem ser parte do método.
   - Hipótese, não testada: com a faixa maior, o canal de recompensa do treino fica comprimido em [0; 0,1], o estado
     distingue menos os históricos e a política depende menos dele. No teste, a normalização é por 5 nos dois casos.
   - Demais métricas do MOPO-tmax (`results/tmax/n4_resumo.csv`, `e6_resumo.csv`). Na regra E6, ele reproduz
     b = 0 também no Length em NX_0 (21,14 ± 0,07 contra 21,15 ± 0,09; MOPO: 20,76 ± 0,09) e na MCD (0,586 contra
     0,563; MOPO: 0,669; dp 0,02): o Length a mais de b = 0 no E6 (+0,39) vem da escala do estado. Na regra N4, o
     FB é igual (Length 27,81 ± 0,07 contra 27,68 ± 0,24: o ciclo continua). O Length em NX_0 diferia com 5
     sementes (18,8 contra 15,9) e empata com 10 (17,57 contra 17,25). O `fb_rotacao.py` não foi rodado nessas
     execuções.
   - O comportamento no FB não muda do MOPO para b = 0 nem para o MOPO-tmax; nele a atribuição ao bônus se
     mantém. Com 10 sementes, o mesmo vale para Length e R_tra em NX_0 na regra N4.
5. **No FB com a regra N4, o bônus reduz o ciclo de forma gradual.** Length 27,7 → 21,6 → 14,3; repetição
   0,73 → 0,53 → 0,31; ciclo puro 0,70 → 0,15 → 0,05 (b = 0, 4 e 8,35). A pergunta em aberto ("por que o *survival
   bonus* impede o ciclo no FB") ganha um fato: o efeito é gradual, não um limiar. O mecanismo continua sem
   explicação. Na regra E6, o sinal é o oposto: a repetição no FB cresce com a dose. Com 10 sementes, cada passo
   sucessivo da dose encurta o FB acima do dp (seção "Sementes 5–9").
6. **Nenhuma dose leva a recompensa por passo acima do acaso.** Na regra N4, o R_each fica entre 0,71 e 0,81 contra
   0,872; para b = 4 e b = 8,35 a diferença está dentro do dp (0,111 e 0,095). Em NX_10, o R_tra fica entre 6,3 e 7,4 contra 8,72,
   acima do dp em todos os pontos.

**Sementes 5–9 na regra N4 e controle inverso (17/11; `results/s10/`).** Motivo: a ressalva do item 1.
- Execuções: b = 0, 1, 2 e 4 (`SEEDS="5 6 7 8 9" roda_bonus.sh N4`), MOPO-tmax (`roda_tmax.sh N4`), MOPO e
  DORL-const (`roda_trajetorias.sh`, que passou a aceitar sementes sem log original). São 35 execuções, todas com 200
  épocas. As sementes 0–4 do MOPO e do DORL-const são as reexecuções `*_traj_s*`, idênticas às originais.
- Controle inverso, sementes 0–4: b = 0 com o `train_max` do MOPO na regra E6 (`TM_SB0_E6_s*`; `SB0-tmin-E6`).
- Regra de leitura fixada antes de rodar, com 10 sementes por ponto no Length em NX_0:
  (a) se b = 0 e MOPO-tmax ficarem a menos de um dp, os 2,9 passos eram ruído e os dois viram uma referência só;
  (b) uma dose intermediária conta como efeito se superar essa referência por mais que o maior dp;
  (c) se MOPO-tmax continuar acima do MOPO e de b = 0, a equivalência entre as duas configurações precisa ser revista.
- Controle inverso: se a hipótese vale, a cobertura volta à do MOPO-E6 (50,2%) e o Length a 20,76.

Resultado, regra N4, 10 sementes por linha (`results/s10/n4_resumo.csv`; cobertura em NX_10, `results/s10/alem/`):

| Configuração | Length (NX_0) | R_tra (NX_0) | R_each (NX_0) | Cobertura (%) | Length (FB) |
|---|---|---|---|---|---|
| MOPO | 16,69 ± 2,19 | 12,39 ± 2,07 | 0,741 ± 0,064 | 9,9 ± 2,7 | 26,12 ± 2,87 |
| MOPO-tmax | 17,57 ± 2,47 | 12,59 ± 2,56 | 0,717 ± 0,101 | 5,6 ± 0,8 | 27,63 ± 0,36 |
| b = 0 | 17,25 ± 2,25 | 12,28 ± 2,02 | 0,713 ± 0,080 | 5,9 ± 0,9 | 27,39 ± 0,50 |
| b = 1 | 18,89 ± 2,13 | 13,58 ± 2,30 | 0,718 ± 0,074 | 4,8 ± 0,7 | 25,41 ± 1,13 |
| b = 2 | 19,34 ± 2,72 | 13,53 ± 2,55 | 0,694 ± 0,051 | 5,0 ± 0,8 | 22,96 ± 1,83 |
| b = 4 | 19,95 ± 2,06 | 15,90 ± 1,64 | 0,801 ± 0,082 | 4,2 ± 0,7 | 19,64 ± 2,28 |
| b = 8,35 (DORL-const) | 23,56 ± 2,87 | 18,64 ± 1,03 | 0,802 ± 0,093 | 3,3 ± 0,3 | 14,72 ± 2,53 |
| Bônus zero (b = 0 e MOPO-tmax, 20) | 17,41 ± 2,31 | 12,44 ± 2,25 | 0,715 ± 0,089 | 5,8 ± 0,8 | 27,51 ± 0,44 |
| Aleatória (5) | 4,60 ± 0,02 | 4,01 ± 0,02 | 0,872 ± 0,001 | 86,5 | 4,59 ± 0,01 |

Leitura pela regra:
1. **(a) vale.** b = 0 − MOPO-tmax = −0,32 passo (maior dp 2,47). A diferença de 5 sementes (−2,9) inverte nas
   sementes 5–9 (18,63 contra 16,35). Os dois formam a referência de bônus zero.
2. **(c) não ocorre.** MOPO-tmax − MOPO = +0,88 (dp 2,47); MOPO − referência = −0,72 (dp 2,31). As três
   configurações sem bônus têm o mesmo Length.
3. **(b): só doses grandes.** Contra a referência: b = 1, +1,48 (maior dp 2,31); b = 2, +1,93 (2,72): dentro.
   b = 4, +2,54 (2,31): supera, por margem pequena. b = 8,35, +6,15 (2,87): supera. Entre doses sucessivas, só
   4 → 8,35 supera o dp. O R_tra segue o mesmo padrão (b = 4: +3,46; b = 8,35: +6,21; b = 1 e 2 dentro).
   - Exploratório, fora da regra: Spearman entre b e o Length nas 50 execuções com b ≤ 4 = 0,42 (p = 0,002). Há
     tendência, mas b = 1 e b = 2 não se distinguem do bônus zero com 10 sementes.

Outros fatos (não estavam na regra):
4. **O FB responde à dose em todos os passos.** Length 27,39 → 25,41 → 22,96 → 19,64 → 14,72 (b = 0 a 8,35); cada passo
   sucessivo supera o maior dp (−1,97, dp 1,13; −2,45, dp 1,83; −3,32, dp 2,28; −4,92, dp 2,53). Repete-se nas duas
   metades (sementes 0–4: 27,7 → 25,9 → 23,6 → 21,6 → 14,3; 5–9: 27,1 → 25,0 → 22,4 → 17,7 → 15,1). É o efeito mais
   robusto do bônus na regra N4.
5. **R_each não muda com a dose.** Contra a referência, nenhuma dose passa do dp em NX_0 (b = 8,35: +0,086, dp 0,093)
   nem em NX_10 (+0,091, dp 0,124). Todas ficam abaixo do acaso (0,872).
6. **Cobertura:** MOPO 9,9 ± 2,7% contra 5,8 ± 0,8% do bônus zero, sem sobreposição entre sementes (7,5–16,3% contra
   4,3–7,2%). Da queda MOPO → DORL-const (6,7 pontos), 4,2 ocorrem com bônus zero (63%). Contra a referência:
   b = 1, −0,92 ponto (maior dp 0,80, supera); b = 2, −0,76 (dentro); b = 4, −1,52; b = 8,35, −2,46 (superam).
7. **Personalização na regra N4 sobe com a dose** (bônus zero 0,35; b = 1: 0,39; b = 2: 0,48; b = 4: 0,53;
   b = 8,35: 0,61; MOPO 0,49; `results/s10/alem/alem_resumo.csv`), ao contrário do E6 (0,83 → 0,16). Cobertura e
   personalização andam em sentidos opostos na regra N4.

Resultado, controle inverso (regra E6, 5 sementes; `results/s10/e6_resumo.csv`, `results/s10/alem/`):

| Configuração | Recompensa | `train_max` | Cobertura (%) | Personalização | MCD (NX_0) | Length (NX_0) |
|---|---|---|---|---|---|---|
| MOPO-E6 | MOPO | 0,94 | 50,2 ± 5,0 | 0,958 | 0,669 ± 0,019 | 20,76 ± 0,09 |
| SB0-tmin-E6 | b = 0 | 0,94 | 48,0 ± 8,9 | 0,945 | 0,670 ± 0,030 | 20,69 ± 0,14 |
| SB0-E6 | b = 0 | 9,48 | 24,3 ± 3,9 | 0,831 | 0,563 ± 0,019 | 21,15 ± 0,09 |
| MOPO-tmax-E6 | MOPO | 9,48 | 22,8 ± 2,7 | 0,794 | 0,586 ± 0,020 | 21,14 ± 0,07 |

8. **O controle inverso confirma a hipótese.** Com a recompensa de b = 0 e o `train_max` do MOPO, a política volta ao
   MOPO em cobertura, personalização, MCD e Length. As sementes não se sobrepõem às de b = 0 (38,1–61,1% contra
   19,9–28,5%). Nos dois sentidos, o que decide é o `train_max`, não a recompensa. O script só aceita a execução se
   o log registra a troca (linha `TMAX_LAMBDA`).

**Limitações.** Sem ponto entre b = 4 e 8,35. Na regra N4, 10 sementes por ponto, exceto b = 0,5 (5); na regra E6,
5. O controle inverso só rodou no E6. Mesmo com 10 sementes, o efeito de b = 4 no Length passa o dp por margem
pequena. λ₁ fixo.
Uma calibração do E6.

## Etapa 5 — Sementes 5–9 com N = 8c

Com 10 sementes, as duas diferenças de Length com N = 8c que as etapas anteriores contavam como efeito ficam dentro do
ruído. Em R_tra, a vantagem do DORL sobre o MBPO se mantém, mas não vem de um único termo.

**Pergunta.** DORL − MBPO (+3,5) e MOPO → DORL-const (+3,0), medidos em Length com 5 sementes e N = 8c (semana 9,
etapa 6; etapa 2 acima), se mantêm com 10 sementes?

**Motivo.** Na regra N4, uma diferença de 2,9 passos medida com 5 sementes sumiu com 10 (etapa 4, "Sementes 5–9"). As
duas diferenças com N = 8c eram da mesma ordem.

**Procedimento.** Sementes 5–9 de MBPO e DORL (`SEEDS="5 6 7 8 9" roda_a5.sh 8`) e de MOPO e DORL-const (idem, com
`METODOS="MOPO DORLconst"`; DORL-const com `PE_CONST=1.9609`), em duas filas na 4090. São 20 execuções, todas com 200
épocas e `num_leave_compute` 8 no log; nenhuma tem a linha `TMAX_LAMBDA`. Agregação com o `e1_protocolos.py` →
`results/s10/n8c_*`, com a aleatória de N = 8c (`results/e3/aleatoria_N8c.json`, 5 sementes). Regra fixada antes de
rodar: a do trabalho (a diferença conta se supera o maior dp entre sementes dos dois métodos), em Length no NX_0.

**Resultado.** NX_0, 10 sementes por método (aleatória: 5); Length (FB) no protocolo com repetição.

| Método | Length | R_tra | R_each | MCD | Length (FB) |
|---|---|---|---|---|---|
| MBPO | 19,43 ± 2,25 | 13,71 ± 2,04 | 0,707 ± 0,063 | 0,335 ± 0,025 | 24,28 ± 1,63 |
| MOPO | 18,65 ± 1,89 | 12,60 ± 2,06 | 0,679 ± 0,075 | 0,317 ± 0,047 | 24,65 ± 1,36 |
| DORL-const | 20,45 ± 1,37 | 14,76 ± 2,17 | 0,723 ± 0,079 | 0,396 ± 0,019 | 10,62 ± 2,92 |
| DORL | 21,62 ± 1,02 | 17,25 ± 0,62 | 0,799 ± 0,037 | 0,391 ± 0,019 | 10,35 ± 1,87 |
| Aleatória (5) | 4,16 ± 0,01 | 3,62 ± 0,01 | 0,871 ± 0,001 | 0,827 ± 0,001 | 4,15 ± 0,01 |

Decomposição em Length (NX_0); a última coluna separa as sementes 0–4 (o resultado anterior) e 5–9.

| Passo | Termo acrescentado | Diferença (maior dp) | Leitura | Sementes 0–4 / 5–9 |
|---|---|---|---|---|
| MBPO → MOPO | P_U | −0,78 (2,25) | Ruído | −0,52 / −1,03 |
| MOPO → DORL-const | *Survival bonus* | +1,80 (1,89) | Ruído | +3,03 / +0,56 |
| DORL-const → DORL | Variação do P_E | +1,17 (1,37) | Ruído | +0,94 / +1,40 |
| MBPO → DORL | Os três | +2,19 (2,25) | Ruído | +3,45 / +0,93 |

Leitura pela regra:
1. **Nenhuma das duas diferenças se confirma.** DORL − MBPO cai de +3,5 para +2,2 (dp 2,3) e MOPO → DORL-const, de
   +3,0 para +1,8 (dp 1,9). Nas sementes 5–9, +0,9 e +0,6. É o padrão da regra N4: diferenças de ~3 passos medidas com
   5 sementes não resistem a 10.
2. **Com N = 8, a vantagem do DORL em Length não se distingue do ruído em nenhuma das duas janelas** (+0,6 com o recorte,
   E3; +2,2 sem ele). A afirmação principal vale com N = 1, 4 e 4c, não com N = 8. O recorte ainda muda o Length do
   DORL (20,1 → 21,6; dp 1,0), e não o do MBPO (19,4 → 19,4).

Outros fatos (não estavam na regra):
3. **Em R_tra, a vantagem do DORL se mantém, sem um termo único.** DORL − MBPO = +3,54 (dp 2,04), acima nas duas
   metades (+4,08 e +3,00). Nos passos: MOPO → DORL-const +2,16 (dp 2,17), no limite; DORL-const → DORL +2,50
   (dp 2,17), pouco acima. Nas métricas do artigo, é a única comparação DORL-const × DORL acima do dp em todo o
   trabalho (seis resumos, três protocolos): numa regra, numa métrica e por margem pequena, que vem das sementes 5–9
   (0–4: +1,65; 5–9: +3,34). Em Length (+1,17), R_each (+0,076, dp 0,079),
   MCD (−0,005) e R_tra no NX_10 (+0,76, dp 0,82), fica dentro do ruído.
4. **O bônus encurta o FB também com N = 8c:** MOPO → DORL-const = −14,0 passos (dp 2,9), nas duas metades (−13,8 e
   −14,3); a variação do P_E não muda o FB (−0,3). Como na regra N4, é o efeito mais robusto do bônus.
5. **Recompensa por passo abaixo do acaso:** R_each de 0,68–0,80 contra 0,871 e R_tra no NX_10 de 6,20–7,31 contra
   8,72, em todos os métodos de RL. DORL − MBPO em R_each = +0,092 (dp 0,063), acima.
6. **MCD:** MOPO → DORL-const = +0,079 (dp 0,047), acima; todos seguem muito abaixo da taxa de base do acaso (0,32–0,40
   contra 0,83).

**Limitações.** Cobertura, personalização e ciclo puro com N = 8c seguem com 5 sementes: dependem das trajetórias, que
ficam na 4090. Uma constante do bônus e um λ₁; a aleatória tem 5 sementes.
