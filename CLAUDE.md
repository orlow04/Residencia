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
| R | Reprodução MBPO/MOPO/DORL, 5 sementes | MBPO feito; DORL rodado, a tabular; MOPO em execução |
| S1 | Diagnóstico do P_E sem treino (`diagnostico_pe.py`) | Feito |
| S2 | r̂ × recompensa de avaliação (`r_hat_vs_avaliacao.py`) | Feito |
| S3 | Política aleatória (`politica_aleatoria.py`) | Feito |
| E1 | Inversão por protocolo (`e1_protocolos.py --aleatoria aleatoria.json`) | Parcial: falta incluir MOPO e DORL |
| P | Registro do P_E no treino (`pe_log.patch`, `analisa_pe_log.py`) | Pendente |
| E2 | Ablação: DORL com P_E constante | Pendente (constante vem de P) |

Resultados principais até aqui (detalhes em `docs/experimentos_semana8.md`):
- MBPO, 5 sementes, NX_0: R_tra 11,69 ± 2,32; R_each 0,736 ± 0,071; Length 15,78 ± 1,83; MCD 0,323 ± 0,061 (artigo: 12,04; 0,770; 15,65; 0,362).
- S1: sob políticas de referência, o P_E explica ~99% da variação da recompensa do DORL; 96,3% dos pares da tabela valem 1.
- S2: top-10 por r̂ = 0,40 de recompensa de avaliação, contra 0,87 do acaso e 2,33 da média do vídeo.
- S3: política aleatória (NX_0) R_tra 4,01, R_each 0,872, Length 4,60, MCD 0,828; supera UCB e ε-greedy.
- E1: com 10 rodadas forçadas (NX_10), o MBPO passa de 1º a último em R_tra.

## Próxima semana (planejada)

1. Registrar as trajetórias de avaliação (itens e categorias por passo): patch novo no coletor/avaliador.
2. Investigar o FB: o MBPO dura ~28 passos; hipótese de rotação de 5+ categorias que nunca dispara a saída.
3. "Bolha": definição operacional + métrica por trajetória sob NX_10, relativa a uma referência.
4. E5: modelo de usuário treinado com `watch_ratio` bruto limitado a 5 (o carregador `KuaiEnv.get_df_kuairec` só lê `watch_ratio_normed`; precisa de alteração) e repetir MBPO e DORL.
5. Esqueleto do relatório e apêndice técnico.
Adiados: E3 (variar N da regra de saída), KuaiRand, mais sementes dos bandits.

## Armadilhas conhecidas

- Não alterar `--training-num`, `--test-num` nem `--episode-per-collect` (quebra o coletor do tianshou).
- O primeiro comando de DORL no README do DORL usa `0.01/0.05`; o que gerou a Tabela 2 é `0.05/5`.
- Filtro de logs por semente sempre estrito: `[[]DORL_s[0-9][]]_*.log` (existiu um `dorl_s.out` de execução com `$s` vazio).
- O `results/versions.txt` antigo capturou o Python do sistema; o registro válido é `pip freeze` com o venv ativo (`versions_venv.txt`).
- `pe_log.patch`: sem `PE_LOG`/`PE_CONST`, a recompensa é idêntica à original (testado). `PE_CONST` só nas execuções `DORLconst_s*`.
- Matrizes do modelo de usuário: `find saved_models -name "*_mat*.pickle"`; forma 1411 × 3327, mesma ordem (LabelEncoder) do ambiente de avaliação.
- O ± da Tabela 2 é entre épocas de uma execução; comparar métodos pelo desvio **entre sementes**.
- Erros já cometidos e corrigidos: (1) "a semente 2 é a mais próxima do artigo nas quatro métricas" é falso — ela é a única a menos de um desvio do artigo nas quatro; (2) a MCD do KuaiRec usa 10 categorias dominantes, não 12 (a Justificativa de Replicação ainda diz 12).

## Arquivos deste repositório

| Arquivo | Função |
|---|---|
| `setup.sh`, `requirements*.txt`, `smoke_test.sh`, `make_synthetic_kuairec.py`, `validate_kuairec.py` | Ambiente e validação dos dados |
| `table2_from_log.py` | Resume um log no formato da Tabela 2 (reproduz os 40 valores do artigo com os logs dos autores) |
| `e1_protocolos.py` | Agregação entre sementes, posições e inversões nos três protocolos |
| `diagnostico_pe.py`, `r_hat_vs_avaliacao.py`, `politica_aleatoria.py` | Análises S1, S2, S3 (CPU, sem treino) |
| `pe_log.patch`, `analisa_pe_log.py` | Registro do P_E no treino e constante do E2 |
| `tests/` | Testes de fidelidade contra cópias literais do código do DORL (ver `tests/README.md`) |
| `results/` | Logs, tabelas e saídas exportadas da 4090 |
| `docs/` | Contexto do TCC, registro de experimentos e figuras |

## Documentos fora do repositório

- Google Drive, pasta da residência: Termo semanal (`GABRIEL ORLOW DE OLIVEIRA SENE_DDMMAA`), "Experimentos em RLRS - Avaliação/Bolhas", "Justificativa de Replicação do DORL", planilha "Artigos Residência".
- Relatórios de referência de residências anteriores: Artur Novais e Daniel Fazzioni (2025).
