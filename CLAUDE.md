# Residência em IA (UFG) — TCC de Gabriel: RL + Sistemas de Recomendação

Contexto detalhado (histórico, literatura, decisões, achados no código): @docs/contexto_tcc.md
Registro completo dos experimentos da semana 8: @docs/experimentos_semana8.md

## Como trabalhar com o Gabriel

- Português, registro acadêmico. Feedback direto e crítico, sem rodeios.
- Respostas curtas e claras: uma resposta por pergunta, sem listas longas de alternativas. Ele já disse que respostas longas ficam confusas.
- Papel do Claude: assistente de pesquisa e revisor crítico. Não escrever partes do texto acadêmico do TCC (o relatório final) sem pedido explícito. Código, scripts, análises, revisão e documentos técnicos pedidos explicitamente: sim.
- Separar sempre fato verificado (com número e arquivo) de hipótese. Não afirmar um resultado que ainda não rodou.
- Antes de afirmar algo sobre o DORL, conferir no código (DORL-codes `818c9bd`), não só no artigo ou no README.

## O TCC em uma página

- Bacharelado em IA, UFG, formato Residência (coordenação: Cedric Luiz de Carvalho). Defesa na primeira semana de dezembro de 2026.
- Cerca de 10 gates semanais. Cada gate tem um Termo de Aceite com **Entrega** (o que foi feito, com links) e **Planejamento** (próxima semana), assinado pelo coordenador.
- Entregável final: relatório narrativo da jornada de especialização + apêndices técnicos. Não é monografia tradicional; avalia especialização e jornada, não novidade científica.
- Especialização declarada: Aprendizado por Reforço + Sistemas de Recomendação (RLRS).
- Pergunta orientadora (adotada no fim de outubro): **"Quais sinais usados são influentes para avaliar um sistema de recomendação baseado em reforço?"** Decomposta em sinais de treino (r̂, P_U, P_E), regras do simulador (regra de saída) e métricas/protocolos de avaliação.
- Direção: mostrar como efeitos do tipo Mateus aparecem, ou são mal medidos, quando boas práticas de avaliação não são seguidas.

## Base experimental

- Artigo replicado: Gao et al. 2023, **DORL** (SIGIR '23). Código: `chongminggao/DORL-codes` no commit `818c9bd`; fork do tianshou no commit `0f59e38` (submódulo).
- Dados: KuaiRec. Treino no big_matrix (12.530.806 interações); avaliação no small_matrix (1.411 × 3.327, totalmente observado).
- Este repositório é o "passo 1": `setup.sh` cria `dorl-workspace/` (DORL-codes + `.venv` Python 3.10). Os comandos do DORL rodam de `dorl-workspace/DORL-codes`; os scripts daqui são chamados como `../../script.py`.
- Hardware: RTX 4090 via SSH (treino das políticas, ~45–50 min por semente; bandits ~8 h 40 min) e MacBook M3 Pro (análises em CPU).
- Argumentos tirados dos **logs dos autores**, não do README do DORL: `--num_leave_compute 4 --leave_threshold 0 --which_tracker avg --reward_handle cat --window_size 3 --read_message pointneg`.

| Método | `--lambda_variance` | `--lambda_entropy` | `--message` |
|---|---|---|---|
| MBPO | 0 | 0 | `MBPO_s{0..4}` |
| MOPO | 0.05 | 0 | `MOPO_s{0..4}` |
| DORL | 0.05 | 5 | `DORL_s{0..4}` |
| DORL com registro do P_E | 0.05 | 5 | `DORL_pelog_s2` (+ `PE_LOG=arquivo.csv`) |
| DORL-const (E2) | 0.05 | 5 | `DORLconst_s{0..4}` (+ `PE_CONST=<c>`) |

## Estado em 29/10/2026 (atualizar a cada gate)

| Id | Experimento | Status |
|---|---|---|
| R | Reprodução MBPO/MOPO/DORL, 5 sementes | Feito |
| S1 | Diagnóstico do P_E sem treino (`diagnostico_pe.py`) | Feito |
| S2 | r̂ × recompensa de avaliação (`r_hat_vs_avaliacao.py`) | Feito |
| S3 | Política aleatória (`politica_aleatoria.py`) | Feito |
| E1 | Inversão por protocolo (`e1_protocolos.py --aleatoria aleatoria.json`) | Feito (7 métodos) |
| P | Registro do P_E no treino (`pe_log.patch`, `analisa_pe_log.py`) | Feito (s2; c = 1,9609) |
| E2 | Ablação: DORL com P_E constante (`DORLconst_s0..4`) | Feito |

Resultados principais (NX_0, média ± dp entre 5 sementes; detalhes em `docs/experimentos_semana8.md` e `results/e1/e1_saida.txt`):

| Método | R_tra | R_each | Length | MCD |
|---|---|---|---|---|
| MBPO | 11,69 ± 2,32 | 0,736 ± 0,071 | 15,78 ± 1,83 | 0,323 ± 0,061 |
| MOPO | 12,00 ± 2,61 | 0,763 ± 0,061 | 15,60 ± 2,34 | 0,366 ± 0,085 |
| DORL | 18,27 ± 2,46 | 0,780 ± 0,086 | 23,42 ± 1,10 | 0,431 ± 0,027 |
| DORL-const | 18,53 ± 0,29 | 0,781 ± 0,095 | 24,00 ± 2,66 | 0,406 ± 0,026 |
| Aleatória | 4,01 ± 0,02 | 0,872 ± 0,001 | 4,60 ± 0,02 | 0,827 ± 0,001 |

- Afirmação principal reproduzida (DORL > MBPO em Length e R_tra, acima do dp); C1 (MOPO × MBPO em MCD) dentro do ruído.
- E2: DORL-const = DORL nas quatro métricas → o ganho do DORL vem do bônus fixo por passo, não da variação do P_E.
- P: na política treinada, P_E ≈ 1,96 de 2, quase constante; no fim do treino as sessões simuladas duram 30 passos. DORL_pelog_s2 = DORL_s2 (patch inerte, execução determinística).
- S1/S2/S3: P_E explica ~99% da variação da recompensa (políticas de referência); top-10 do r̂ = 0,40 contra 0,87 do acaso; bandits abaixo do acaso.
- E1: em NX_10 os quatro métodos de RL ficam abaixo do acaso e dos bandits em R_tra; em R_each, abaixo do acaso nos três protocolos; no FB, DORL e DORL-const duram ~15 passos contra ~27 de MBPO/MOPO.
- Logs do DORL: havia repetições e execuções incompletas; o `e1_protocolos.py` usa só o log completo mais recente de cada semente.

## Semana 9 (gate em 10/11/2026) — registro em `docs/semana9.md`

| Id | Experimento | Status |
|---|---|---|
| Traj | Registro das trajetórias de avaliação (`traj_log.patch`); 20 políticas reexecutadas, idênticas aos logs | Feito |
| Bolha | Variedade de categorias em 10 itens (NX_10) contra o usuário (8,02) e o acaso (8,88): RL 9,3–10,2, sem bolha | Feito |
| E5 | Modelo de usuário com watch_ratio bruto ≤ 5: MBPO muda (R_each 0,736 → 1,034), DORL não | Feito |
| E3/A5 | N = 1, 4, 8, janela corrigida: DORL − MBPO em Length +4,8 / +6,7 / +3,5 (N = 8 original: +0,6, artefato do recorte) | Feito |
| FB | MBPO cicla; nos ciclos puros (38–61% das trajetórias) o bloco tem N + 1 itens em 80–99% (N = 1, 4) | Feito |
| Além | Cobertura de itens: DORL 3,3%, MBPO 12%, aleatória 86,5% — inverte a leitura da MCD | Feito |
| Sessões | P(fim | repete categoria) 6,3% contra 2,6% nas sessões reais; o simulador supõe 100% contra 0% | Feito |
| E6 | Saída probabilística calibrada (N = 1, 0,063/0,026): vantagem do DORL some (Length +0,3), RL ≈ acaso; cobertura MBPO-E6 52,8% × DORL-E6 2,7% (personalização 0,15); DORL-const-E6 = DORL-E6 (2,5%, 0,16) e MOPO-E6 = MBPO-E6 (50,2%, 0,96) → a concentração vem do bônus fixo por passo, não da variação do P_E nem do P_U | Feito |

**Enquadramento adotado (10/11):** o protocolo de avaliação é estruturado pelo capítulo de avaliação de Aggarwal
(objetivos do desenho de avaliação → concentração em vários níveis; desenho da avaliação offline → sinal de treino ×
avaliação; limitações das medidas → a política explora a regra do simulador), complementado por controles de RL
(Sutton & Barto: média de várias execuções → desvio entre sementes; papel da recompensa → ablação por constante;
Afsar et al.: dependência de simuladores → referência sem sinal). Não usar referenciais que o Gabriel não estudou.
Nomes das seções do Aggarwal foram dados de memória: confirmar no livro.

## Semana 10 — registro em `docs/semana10.md`; síntese em `docs/sintese.md`; números do Termo final em `docs/resultados_semana10.md`

- Revisitas (feito): as políticas repetem 6,7–74,3% dos passos FB contra 3,4% nas sessões reais; a cobertura sem
  repetições é 1,1–2,6× maior; a repetição real segue a popularidade (top 10%: 20,3% contra 10,6%), a das políticas não.
- Janela corrigida para MOPO e DORL-const (feito, 12/11): com N = 4c/8c, só MOPO → DORL-const passa do dp
  (Length +5,7/+3,0; cobertura 10,4% → 3,3%); P_U e variação do P_E dentro do ruído. MOPO cicla como o MBPO no FB.
- Bandits com 5 sementes (feito, 13/11): abaixo do acaso em R_tra e Length; UCB empata com o acaso em R_each (NX_0) e
  supera MBPO/MOPO/DORL nela; em NX_10, RL abaixo dos bandits só para MBPO/MOPO (DORL/DORL-const: ruído).
- Termo técnico: o bônus fixo por passo do DORL é um *survival bonus* (usar o nome em inglês).
- Varredura do *survival bonus* (feito, 14/11; etapa 4): na regra N4 o Length só sobe com doses grandes (ver abaixo) e o
  ciclo no FB diminui (ciclo puro 0,70 → 0,15 → 0,05); no E6 a cobertura cai com b (24,3% → 9,9% → 2,5%) sem ganho de
  Length. **b = 0 não reproduz o MOPO em cobertura** (E6: 50,2% → 24,3%; N4: 11,2% → 6,1%): MOPO → DORL-const muda
  também `train_max` (normalização da recompensa no estado, `run_Policy_Main.py:250`). Causa testada em 16/11
  (`MOPO-tmax`, `roda_tmax.sh`): só com o `train_max` do DORL, a cobertura do MOPO cai para a de b = 0 (E6: 22,8%;
  N4: 5,4%). DORL × MOPO tem três fatores: variação do P_E, *survival bonus* e escala da recompensa no estado.
- Sementes 5–9 na regra N4 e controle inverso (feito, 17/11; `results/s10/`): com 10 sementes, MOPO, MOPO-tmax e b = 0
  empatam no Length (16,7–17,6); contra o bônus zero (17,4 ± 2,3), b = 1 e 2 ficam dentro do dp, b = 4 supera por
  pouco (+2,5) e 8,35 com folga (+6,2). O FB encurta em todos os passos da dose (27,4 → 14,7). Controle inverso:
  b = 0 com o `train_max` do MOPO volta à cobertura do MOPO no E6 (48,0% × 50,2%). Hipótese do `train_max` confirmada
  nos dois sentidos.
- Avaliação off-policy no log real (feito, 16/11; `docs/off_policy.md`; no `main` desde 17/11, merge `d706527`):
  SNIPS com π_β por contagens em 9,45 M passos de usuários fora da avaliação, escolhido em validação semi-sintética;
  20 políticas retreinadas com checkpoint (`roda_ckpt.sh`, época 200, logs conferidos contra os originais). 95% da
  exposição em 2–16 itens (Gini 0,997–1,000 contra 0,244 da política de registro); 17 de 20 abaixo da política de
  registro (0,881); ESS 215–2.911; ordem dos métodos não identificável (dp entre sementes maior que as diferenças).

## Próximos passos (a semana 10 é a última; no Termo final, o campo Planejamento leva os próximos passos, como nos relatórios de referência)

1. Feito: E6 com DORL-const e MOPO (11/11), revisitas (11/11), tabela de síntese (`docs/sintese.md`); a avaliação
   off-policy entrou no `main` (17/11, merge `d706527`) e entra no relatório (síntese B8; controle g do protocolo).
2. Na fila da 4090 desde 17/11 (`a5_fila5.out`, `a5_fila6.out`): sementes 5–9 com N = 8c (MBPO/DORL e MOPO/DORLconst)
   para DORL − MBPO (+3,5) e MOPO → DORL-const (+3,0), hoje com 5 sementes. Conferir com
   `grep -c ": fim" a5_fila5.out a5_fila6.out` (10 em cada, sem "INCOMPLETO"); exportar e ler com a mesma regra.
3. Depois do N = 8c: congelar os experimentos (tag).
4. Relatório e apêndice técnico: esperam os modelos que o Gabriel vai receber; não propor esqueleto antes.
Em aberto: Length do MBPO não monótono no N; por que o *survival bonus* impede o ciclo no FB (o ciclo é dos métodos sem bônus);
por que a escala da recompensa no estado reduz a cobertura (hipótese: estado menos informativo); por que, na regra N4, a personalização sobe com a dose enquanto a cobertura cai.

## Armadilhas conhecidas

- Não alterar `--training-num`, `--test-num` nem `--episode-per-collect` (quebra o coletor do tianshou).
- O primeiro comando de DORL no README do DORL usa `0.01/0.05`; o que gerou a Tabela 2 é `0.05/5`.
- Filtro de logs por semente sempre estrito: `[[]DORL_s[0-9][]]_*.log` (existiu um `dorl_s.out` de execução com `$s` vazio).
- O `results/versions.txt` antigo capturou o Python do sistema; o registro válido é `pip freeze` com o venv ativo (`versions_venv.txt`).
- `pe_log.patch`: sem `PE_LOG`/`PE_CONST`, a recompensa é idêntica à original (testado). `PE_CONST` só nas execuções `DORLconst_s*`.
- Matrizes do modelo de usuário: `find saved_models -name "*_mat*.pickle"`; forma 1411 × 3327, mesma ordem (LabelEncoder) do ambiente de avaliação.
- O ± da Tabela 2 é entre épocas de uma execução; comparar métodos pelo desvio **entre sementes**.
- Pares usuário–vídeo repetidos no big_matrix (11,8% das linhas dos usuários de avaliação): 56% das repetições
  são duplicatas exatas (mesmo timestamp e watch_ratio). Sem elas, 5,5% das visualizações repetem um par, 70% em
  outra sessão (`revisitas.py`). Não tratar como revisitas sem filtrar.
- `e1_protocolos.py --out X` é **prefixo**, não pasta: usar `--out ../../results/a5/e1` (gera `results/a5/e1_*.csv`).
  O `exporta_resultados.sh` copia `e1_*` do DORL-codes para `results/e1/`: não deixar `e1_saida.txt` velho lá
  (os antigos estão em `DORL-codes/e1_antigo/` na 4090).
- Push da 4090 falhando com `vscode-git-*.sock ECONNREFUSED`: terminal de sessão antiga do VS Code. Abrir terminal novo
  ou, no Mac, `git pull <host-4090>:dorl_passo1 main`. Não deixar credencial do GitHub na 4090 (máquina compartilhada).
- Os usuários de avaliação não têm nenhuma interação com os itens do small_matrix no big_matrix (0 de 533.090 linhas);
  os demais usuários (até 5.765) têm 9,45 M linhas com esses itens (base da avaliação off-policy, `ope_dados.py`).
- As políticas não são salvas por padrão (`--is_save` desligado); o estado não usa embedding de usuário. Os checkpoints
  da época 200 (`CK_*`, `roda_ckpt.sh`) ficam na 4090, em `saved_models/KuaiEnv-v0/A2C_with_emb/`.
- FB: o bloco de N + 1 itens vale só entre as trajetórias em ciclo puro, não para a média de itens distintos.
- Ruído do Length na regra N4: `MOPO-tmax` e SB0 são a mesma configuração a menos de 0,0005 por passo; com 5 sementes
  diferiam em 2,9 passos (18,8 × 15,9), com 10 empatam (17,6 × 17,2) e a diferença inverte nas sementes 5–9
  (`results/s10/`). Com 5 sementes, diferença de Length de ~3 passos não é confiável, mesmo acima do dp. Ainda com
  5 sementes e dessa ordem: DORL − MBPO com N = 8c (+3,5); MOPO → DORL-const com N = 8c (+3,0).
- Erros já cometidos e corrigidos: (1) "a semente 2 é a mais próxima do artigo nas quatro métricas" é falso — ela é a única a menos de um desvio do artigo nas quatro; (2) a MCD do KuaiRec usa 10 categorias dominantes, não 12 (a Justificativa de Replicação ainda diz 12); (3) "MOPO → DORL-const acrescenta só o bônus fixo" é falso — muda também a normalização do estado; vale para Length e FB, não para a cobertura (54–65% da queda ocorre com bônus zero; confirmado pelo `MOPO-tmax`).

## Arquivos deste repositório

| Arquivo | Função |
|---|---|
| `setup.sh`, `requirements*.txt`, `smoke_test.sh`, `make_synthetic_kuairec.py`, `validate_kuairec.py` | Ambiente e validação dos dados |
| `table2_from_log.py` | Resume um log no formato da Tabela 2 (reproduz os 40 valores do artigo com os logs dos autores) |
| `e1_protocolos.py` | Agregação entre sementes, posições e inversões nos três protocolos |
| `diagnostico_pe.py`, `r_hat_vs_avaliacao.py`, `politica_aleatoria.py` | Análises S1, S2, S3 (CPU, sem treino) |
| `pe_log.patch`, `analisa_pe_log.py` | Registro do P_E no treino e constante do E2 |
| `verifica_E2.sh` | Prova que o E2 usou `PE_CONST=1.9609` (repete 2 épocas por semente; o treino é determinístico) |
| `exporta_resultados.sh` | Na 4090: copia para `results/` logs, saídas do E1, resumo do P_E, diff e versões |
| `traj_log.patch`, `verifica_traj.sh`, `roda_trajetorias.sh` | Registro das trajetórias de avaliação (`TRAJ_LOG`), sua verificação e as reexecuções conferidas contra os logs originais |
| `e5_alvo.patch`, `roda_e5.sh` | E5: modelo de usuário com watch_ratio bruto limitado a 5 (`E5_RAW_WR`, `--message rawwr`) e MBPO/DORL com ele (`E5_{MBPO,DORL}_s*`) |
| `roda_e3.sh` | E3: MBPO/DORL com outro N da regra de saída (`--num_leave_compute`, treino e avaliação; `E3N{N}_{MBPO,DORL}_s*`); aleatória com o mesmo N em `results/e3/aleatoria_N*.json` |
| `a5_janela.patch`, `roda_a5.sh` | Janela da regra de saída corrigida (`A5_FIX=1`: min(t, N) anteriores, sem o recorte); MBPO/DORL `A5N{N}_*`; aleatória com `--janela_correta` em `results/e3/aleatoria_N*c.json` |
| `alem_da_acuracia.py` | Critérios além dos do artigo nas trajetórias NX_10 (cobertura, Gini da exposição, novidade, serendipidade, personalização) e inversões entre critérios |
| `mecanismo_bolha.py`, `fb_rotacao.py` | Origem da diversidade (regra de saída contrafactual; V esperado dado o nº de itens dominantes) e estrutura das trajetórias FB |
| `sessoes_reais.py` | Regra de saída × sessões reais do big_matrix (só cadeias contíguas): P(fim de sessão | repete categoria) → `results/sessoes/` |
| `e6_saida_prob.patch`, `roda_e6.sh` | E6: saída probabilística calibrada nas sessões reais (`E6_P_REP=0.063 E6_P_BASE=0.026`, N = 1); MBPO/MOPO/DORL/DORL-const `E6_*`; aleatória com `--p_saida` em `results/e6/aleatoria_E6.json` |
| `revisitas.py` | Revisualização: revisitas reais no big_matrix (sem duplicatas), loop no small_matrix e repetição das políticas no FB (`--traj`) → `results/revisitas/` |
| `roda_bandits.sh` | Sementes adicionais de UCB e ε-greedy (`UCB_s{n}`, `epsilon-greedy_s{n}`) |
| `roda_bonus.sh` | Varredura do *survival bonus*: DORL-const com b = λ₂·(c − 0,291) ∈ {0; 0,5; 1; 2; 4}, regras N4 e E6 (`SB{b}_{N4,E6}_s*`); E1 em `results/sb/` |
| `tmax.patch`, `roda_tmax.sh` | Controle da varredura: MOPO com o `train_max` do DORL (`TMAX_LAMBDA=5`; `TM_MOPO_{N4,E6}_s*`) e, opcional, SB0 com o do MOPO (`TM_SB0_*`); nas tabelas, `MOPO-tmax` e `SB0-tmin` |
| `bolha.py` | Bolha por trajetória sob NX_10 (variedade e Gini-Simpson relativos às janelas de 10 itens do usuário no big_matrix) |
| `figuras.py` | Refaz `docs/figuras/` a partir de `results/` |
| `ope.py`, `ope_contexto.py` | Avaliação off-policy por passo (IPS, SNIPS, DM, DR) e validações semi-sintéticas, com π_β conhecida e dependente do contexto → `results/ope/ope_validacao.txt`, `ope_contexto.txt` |
| `ope_dados.py` | Log real do OPE: sessões dos usuários fora da avaliação com itens do small_matrix (9,45 M passos) e π_β por contagens → `results/ope/ope_dados.txt` (`ope_log.npz`, 206 MB, não versionado) |
| `roda_ckpt.sh` | Retreino de MBPO/MOPO/DORL/DORL-const com checkpoint (`CK_*_s*`), conferido contra os logs originais; mantém só a época 200 |
| `ope_politicas.py`, `ope_compara.py` | Aplica os checkpoints ao log real (SNIPS com IC por *bootstrap*, ESS, concentração da exposição) e cruza com o simulador → `results/ope/` |
| `tests/` | Testes de fidelidade contra cópias literais do código do DORL (ver `tests/README.md`) |
| `results/` | Logs, tabelas e saídas exportadas da 4090; índice em `results/README.md` |
| `docs/` | Contexto do TCC, registro de experimentos e figuras |

## Documentos fora do repositório

- Google Drive, pasta da residência: Termo semanal (`GABRIEL ORLOW DE OLIVEIRA SENE_DDMMAA`), "Experimentos em RLRS - Avaliação/Bolhas", "Justificativa de Replicação do DORL", planilha "Artigos Residência".
- Relatórios de referência de residências anteriores: Artur Novais e Daniel Fazzioni (2025); cópias locais em `papers/`
  (fora do git). No último Termo (10º gate), os dois mantêm o campo Planejamento com 2–3 próximos passos além da
  residência, e a Entrega abre com um resumo da jornada.
