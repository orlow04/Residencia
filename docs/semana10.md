# Semana 10 — últimos experimentos e protocolo de avaliação

Registro iniciado em 11/11/2026. Mesma configuração e mesmos critérios da semana 9 (`docs/semana9.md`): média ± dp
entre 5 sementes; uma diferença conta quando supera o maior dp entre sementes dos dois métodos.

| Id | Experimento | Status |
|---|---|---|
| Rev | Revisualização de vídeos: referência real e repetição das políticas no FB (`revisitas.py`) | Feito (A, B, C) |
| A5-2 | Janela corrigida (N = 4, 8) para MOPO e DORL-const (`roda_a5.sh`) | Em execução (desde 11/11 18:38) |
| Band | Sementes 1–4 de UCB e ε-greedy (`roda_bandits.sh`) | Em execução (desde 11/11 18:38) |
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
`revisitas.py` a partir da reexecução.

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
   - Só a saída calibrada sem bônus fixo (MBPO-E6, MOPO-E6) chega perto. Com a regra determinística, nenhuma
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
5. **Diferença sem interpretação firme.** DORL 13,2% contra DORL-const 30,5% de passos no top 10% (dp 0,108).
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
- MOPO-N4c/N8c e DORL-const-N4c/N8c entram na reexecução da parte C após o A5-2.

## Etapa 2 — Janela corrigida para MOPO e DORL-const

Em execução. Após o término: `e1_protocolos.py` no conjunto do E3, `bolha.py`, `fb_rotacao.py`,
`alem_da_acuracia.py` e a parte C do `revisitas.py`. Atualiza a etapa 5/6 da semana 9 e as linhas A1/B3 da síntese.

## Etapa 3 — Sementes dos bandits

Em execução (~17 h). Após o término: `e1_protocolos.py` com os logs `[UCB]_*`/`[UCB_s*]_*` e
`[epsilon-greedy]_*`/`[epsilon-greedy_s*]_*`. Atualiza S3/E1 e a linha A5 da síntese.
