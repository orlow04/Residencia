# Semana 10 — últimos experimentos e protocolo de avaliação

Registro iniciado em 11/11/2026. Mesma configuração e mesmos critérios da semana 9 (`docs/semana9.md`): média ± dp
entre 5 sementes; uma diferença conta quando supera o maior dp entre sementes dos dois métodos.

| Id | Experimento | Status |
|---|---|---|
| Rev | Revisualização de vídeos: referência real e repetição das políticas no FB (`revisitas.py`) | Feito (A, B, C) |
| A5-2 | Janela corrigida (N = 4, 8) para MOPO e DORL-const (`roda_a5.sh`) | Feito |
| Band | Sementes 1–4 de UCB e ε-greedy (`roda_bandits.sh`) | Feito |
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
duplicatas, sobram 5,5% de repetições, 70% delas em outra sessão. O CLAUDE.md citava 62% de duplicatas, com outro
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
  das políticas. Hipótese não verificada: o big_matrix desses usuários evita os itens do small_matrix.
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

**Interpretação.**

1. **O E2 se reproduz com a janela corrigida.** Nos três cenários, o único passo acima do ruído é o *survival bonus*.
   Ele explica toda a vantagem do DORL sobre o MBPO em Length (+6,7 com N = 4c; +3,5 com N = 8c). Em R_tra, com
   N = 8c, DORL − DORL-const = +1,6, dentro do dp (2,0).
2. **A concentração de itens também vem do bônus.**
   - A cobertura cai de 10,4% para 3,3% (N = 4c) e de 5,7% para 3,8% (N = 8c) no passo MOPO → DORL-const.
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
`results/e3/aleatoria_N{4,8}c.json` e não entra no `e1_saida.txt` do A5.

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
