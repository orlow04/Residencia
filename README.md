# Auditoria da avaliação do DORL no KuaiRec

Repositório do TCC de Gabriel Orlow de Oliveira Sene (Bacharelado em Inteligência Artificial, UFG, formato
Residência, 2026). O trabalho reproduz o DORL (Gao et al., SIGIR '23) no KuaiRec e examina quais sinais decidem o
resultado da avaliação de um sistema de recomendação baseado em reforço (RLRS).

**Pergunta orientadora:** quais sinais usados são influentes para avaliar um sistema de recomendação baseado em
reforço? Ela é decomposta em sinais de treino (a predição do modelo de usuário r̂, a penalidade de incerteza P_U e o
termo de entropia P_E), regras do simulador (a regra de saída do usuário simulado) e métricas e protocolos de
avaliação.

## Resultados principais

Critério em todo o trabalho: média ± dp entre sementes; uma diferença conta quando supera o maior dp entre sementes
dos dois métodos. A síntese completa, com a força de cada evidência, está em [`docs/sintese.md`](docs/sintese.md).

1. **A vantagem do DORL é de duração da sessão, não de recompensa por passo.** Ela depende da regra de saída
   determinística: some com a saída calibrada nas sessões reais, fica dentro do ruído com N = 8 e se inverte quando o
   comprimento da sessão é fixado.
2. **O ganho do DORL vem de um *survival bonus*,** o bônus fixo por passo que o termo de entropia soma à recompensa,
   e não da variação do termo: o DORL com o termo constante (DORL-const) reproduz o DORL.
3. **A comparação DORL × MOPO mistura três fatores:** a variação do termo de entropia (sem efeito), o *survival
   bonus* e a escala da recompensa no estado da política (`train_max`), que muda junto com o peso da entropia.
4. **A concentração depende do nível medido.** Pela métrica do artigo (categorias dominantes), os métodos de RL
   parecem menos concentrados que o acaso; pela cobertura de itens, DORL e DORL-const são os mais concentrados. No
   log real (avaliação off-policy), todas as políticas põem 95% da exposição em 2 a 16 itens.
5. **Os métodos de RL ficam abaixo do acaso em recompensa por passo** nos três protocolos da regra original, e os
   bandits ficam abaixo da política aleatória em recompensa por trajetória e em duração.
6. **Cinco sementes não bastam para diferenças pequenas:** três diferenças de ~3 passos de comprimento, acima do dp
   com 5 sementes, ficaram dentro do ruído com 10: entre duas configurações equivalentes na regra original, e
   DORL − MBPO e MOPO → DORL-const com N = 8 e a janela corrigida.

Os números de cada item estão em [`docs/resultados_semana10.md`](docs/resultados_semana10.md) e nos registros
semanais listados abaixo.

## Como conferir

1. **Ler.** Cada número dos documentos em `docs/` aponta para um arquivo em `results/` (índice em
   [`results/README.md`](results/README.md)). Os logs de todas as execuções, inclusive as descartadas, estão em
   `results/logs/`.
2. **Recalcular em CPU, sem GPU e sem os dados** (precisa de numpy, pandas e scipy; ~20 s):
   ```bash
   python verifica_resultados.py
   ```
   Refaz, a partir dos logs, os resumos entre sementes, os resumos no formato da Tabela 2 e a comparação OPE ×
   simulador, e compara com os arquivos commitados. Confere também que as reexecuções (com trajetórias e com
   checkpoint) reproduzem as execuções originais. Com o ambiente instalado (`setup.sh`), os testes de fidelidade
   conferem os scripts e os patches contra cópias literais do código do DORL (~1 min):
   ```bash
   bash tests/roda_testes.sh
   ```
3. **Refazer do zero, com GPU:** [`docs/reproducao.md`](docs/reproducao.md) (ambiente, dados, modelo de usuário e
   cada experimento).

## Base experimental

| Item | Valor |
|---|---|
| Artigo | Gao et al., SIGIR '23 (DORL) |
| Código | `chongminggao/DORL-codes` no commit `818c9bd`; fork do tianshou no commit `0f59e38`. Clonados pelo `setup.sh`; não são versionados aqui |
| Dados | KuaiRec (Zenodo, registro 18164998; SHA-256 em `results/KuaiRec.zip.sha256`). Treino no big_matrix (12.530.806 interações); avaliação no small_matrix (1.411 usuários × 3.327 itens, totalmente observado) |
| Hardware | RTX 4090 (treino das políticas, ~45–50 min por semente) e MacBook M3 Pro (análises em CPU) |
| Argumentos | Tirados dos logs dos autores, não do README do DORL: `--num_leave_compute 4 --leave_threshold 0 --which_tracker avg --reward_handle cat --window_size 3 --read_message pointneg` |
| Sementes | 0–4 por método; 0–9 na varredura do *survival bonus* na regra original e com N = 8 e a janela corrigida |

| Método | `--lambda_variance` (λ₁) | `--lambda_entropy` (λ₂) | O que é |
|---|---|---|---|
| MBPO | 0 | 0 | Sem penalidades |
| MOPO | 0.05 | 0 | Penalidade de incerteza |
| DORL | 0.05 | 5 | Incerteza e termo de entropia |
| DORL-const | 0.05 | 5 | DORL com o P_E constante (`PE_CONST=1.9609`): só o *survival bonus* |
| UCB, ε-greedy | — | — | Bandits do artigo |
| Aleatória | — | — | Referência sem sinal (`politica_aleatoria.py`, reimplementação validada) |

Protocolos de avaliação do DORL: **FB** (repetição de itens permitida), **NX_0** (repetição proibida; o da Tabela 2)
e **NX_10** (repetição proibida e sessão fixa em 10 passos). Métricas do artigo: R_tra (recompensa por trajetória),
R_each (recompensa por passo), Length (comprimento da sessão) e MCD (fração dos itens em categorias dominantes).
Métricas acrescentadas: cobertura de itens, Gini da exposição, personalização, variedade por usuário (bolha) e
repetição.

## Experimentos

| Id | Pergunta | Scripts | Resultados | Registro |
|---|---|---|---|---|
| R | As afirmações da Tabela 2 se mantêm entre sementes? | `table2_from_log.py`, `e1_protocolos.py` | `results/table2/`, `results/e1/` | `docs/experimentos_semana8.md` |
| S1 | Quanto do P_E é sinal e quanto é preenchimento? | `diagnostico_pe.py` | `results/analises/pe_diag.json` | idem |
| S2 | O r̂ prevê a recompensa de avaliação? | `r_hat_vs_avaliacao.py` | `results/analises/rhat_aval.json` | idem |
| S3 | Quanto vale cada métrica sem nenhum sinal? | `politica_aleatoria.py` | `results/e1/aleatoria.json` | idem |
| E1 | A ordem dos métodos depende do protocolo? | `e1_protocolos.py` | `results/e1/` | idem |
| P | Que valores o P_E assume no treino? | `pe_log.patch`, `analisa_pe_log.py` | `results/pe/` | idem |
| E2 | O ganho do DORL vem da variação do P_E ou do seu valor médio? | `pe_log.patch` (`PE_CONST`), `verifica_E2.sh` | `results/e1/`, `results/pe/verifica_E2.txt` | idem |
| Traj | Registro das trajetórias de avaliação | `traj_log.patch`, `roda_trajetorias.sh`, `verifica_traj.sh` | `results/trajetorias/` | `docs/semana9.md`, etapa 2 |
| Bolha | As políticas formam bolha por usuário? | `bolha.py` | `results/bolha/` | etapa 3 |
| E5 | E com o modelo de usuário treinado no sinal da avaliação? | `e5_alvo.patch`, `roda_e5.sh` | `results/e5/` | etapa 4 |
| E3, A5 | Quanto o resultado depende do N e da janela da regra de saída? | `roda_e3.sh`, `a5_janela.patch`, `roda_a5.sh` | `results/e3/`, `results/a5/`, `results/s10/n8c_*` | etapas 5 e 6; `docs/semana10.md`, etapas 2 e 5 |
| Mecanismo | De onde vêm o ciclo no FB e a diversidade? | `fb_rotacao.py`, `mecanismo_bolha.py` | `results/fb/`, `results/mecanismo/` | etapa 7 |
| Além | Critérios além dos do artigo mudam a leitura? | `alem_da_acuracia.py` | `results/alem/` | etapa 8 |
| Sessões | A regra de saída corresponde às sessões reais? | `sessoes_reais.py` | `results/sessoes/` | etapa 9 |
| E6 | O que sobrevive a uma saída calibrada nas sessões reais? | `e6_saida_prob.patch`, `roda_e6.sh` | `results/e6/` | etapa 10 |
| Revisitas | Como a repetição real se compara à das políticas? | `revisitas.py` | `results/revisitas/` | `docs/semana10.md`, etapa 1 |
| Bandits | Os resultados dos bandits se mantêm com 5 sementes? | `roda_bandits.sh` | `results/e1/` | etapa 3 |
| SB, S10 | O efeito do *survival bonus* cresce com o seu tamanho? | `roda_bonus.sh` | `results/sb/`, `results/s10/` | etapa 4 |
| TM | A escala da recompensa no estado muda a concentração? | `tmax.patch`, `roda_tmax.sh` | `results/tmax/`, `results/s10/` | etapa 4 |
| OPE | Fora do simulador, as conclusões se mantêm? | `roda_ckpt.sh`, `ope*.py` | `results/ope/` | `docs/off_policy.md` |
| Protocolo | Como avaliar um RLRS a partir desses achados? | — | — | `docs/protocolo_avaliacao.md`, `docs/sintese.md` |

## Mapa do repositório

**Documentos (`docs/`), em ordem de leitura sugerida**

| Arquivo | Conteúdo |
|---|---|
| `sintese.md`, `tabelas/sintese.tex` | Afirmações do artigo e achados deste trabalho: sinal examinado, controle, evidência e força |
| `resultados_semana10.md` | Números da última semana: E6, janela corrigida, bandits, repetição, *survival bonus*, `train_max`, OPE |
| `experimentos_semana8.md` | Semana 8: reprodução, S1–S3, E1, P e E2 |
| `semana9.md` | Semana 9: trajetórias, bolha, E5, E3/A5, mecanismo, critérios além do artigo, sessões reais e E6 |
| `semana10.md` | Semana 10: revisitas, janela corrigida para MOPO e DORL-const, bandits, *survival bonus*, `train_max` e sementes 5–9 com N = 8c |
| `off_policy.md`, `off_policy_leitura.md` | Avaliação off-policy no log real; guia de leitura dos estimadores |
| `protocolo_avaliacao.md` | Protocolo de avaliação de RLRS: objetivos O1–O6 e controles (a)–(g) |
| `contexto_tcc.md` | Histórico, decisões e detalhes do DORL verificados no código |
| `reproducao.md` | Reprodução passo a passo |
| `figuras/` | Gráficos, refeitos por `figuras.py` a partir de `results/` |

**Código**

| Grupo | Arquivos |
|---|---|
| Ambiente e dados | `setup.sh`, `requirements.txt`, `requirements-macos.txt`, `smoke_test.sh`, `make_synthetic_kuairec.py`, `validate_kuairec.py` |
| Patches no DORL-codes, cada um inerte sem a sua variável de ambiente | `pe_log.patch`, `traj_log.patch`, `e5_alvo.patch`, `a5_janela.patch`, `e6_saida_prob.patch`, `tmax.patch` |
| Execuções na GPU | `roda_trajetorias.sh`, `roda_e3.sh`, `roda_e5.sh`, `roda_a5.sh`, `roda_e6.sh`, `roda_bandits.sh`, `roda_bonus.sh`, `roda_tmax.sh`, `roda_ckpt.sh`; conferências `verifica_E2.sh` e `verifica_traj.sh`; exportação `exporta_resultados.sh` |
| Agregação dos logs | `table2_from_log.py` (reproduz os 40 valores da Tabela 2 a partir dos logs dos autores), `e1_protocolos.py` |
| Análises em CPU | `diagnostico_pe.py`, `r_hat_vs_avaliacao.py`, `politica_aleatoria.py`, `analisa_pe_log.py`, `bolha.py`, `alem_da_acuracia.py`, `mecanismo_bolha.py`, `fb_rotacao.py`, `sessoes_reais.py`, `revisitas.py`, `figuras.py` |
| Avaliação off-policy | `ope.py`, `ope_contexto.py`, `ope_dados.py`, `ope_politicas.py`, `ope_compara.py` |
| Verificação | `verifica_resultados.py`, `tests/` (ver `tests/README.md`) |

Cada script tem, no cabeçalho, o que faz, o uso e as opções.

## O que fica fora do repositório

| Item | Onde fica | Registro aqui |
|---|---|---|
| Código do DORL e do tianshou | Clonado pelo `setup.sh` | `results/ambiente/dorl_commit.txt`, `dorl_codes_diff.patch` |
| Dados do KuaiRec | Zenodo | `results/KuaiRec.zip.sha256`, `results/kuairec_validation.json` |
| Matrizes do modelo de usuário (~38 MB cada) | RTX 4090 | `results/ambiente/mats.sha256` |
| Trajetórias de avaliação (CSV) | RTX 4090 | `results/trajetorias/*.ok` (SHA-256 de cada CSV conferido) |
| Registro bruto do P_E (~2 GB) | RTX 4090 | `results/pe/pe_log_DORL_s2.sha256` |
| Log do OPE (`ope_log.npz`, 206 MB) e checkpoints das políticas | RTX 4090 | `results/ope/ope_dados.txt`; logs do retreino (`CK_*`) idênticos aos originais (`verifica_resultados.py`) |

## Cuidados na reprodução

- O primeiro comando de DORL no README do DORL usa `--lambda_variance 0.01 --lambda_entropy 0.05`; o que gerou a
  Tabela 2 é `0.05` e `5`.
- Não altere `--training-num`, `--test-num` nem `--episode-per-collect`: valores diferentes de 100 quebram o coletor
  do tianshou.
- O ± da Tabela 2 é a dispersão entre épocas de uma execução; os métodos são comparados pelo desvio entre sementes.
- Há logs repetidos e incompletos. O `e1_protocolos.py` usa o log completo mais recente de cada semente e lista os
  descartados; filtre os logs por semente com glob estrito (`[[]DORL_s[0-9][]]_*.log`).
- O treino é determinístico (`DORL_pelog_s2` = `DORL_s2`): reexecutar uma semente reproduz o log.
- O big_matrix tem pares usuário–vídeo duplicados (mesmo timestamp e watch_ratio); remova-os antes de contar
  revisitas.
- Os usuários de avaliação não têm interação com os itens do small_matrix no big_matrix; a avaliação off-policy usa
  os demais usuários.

## Limitações

- Um conjunto de dados (KuaiRec) e um simulador com regra de saída artesanal, sem usuários reais.
- λ₁ e λ₂ fixos nos valores do artigo; uma única calibração da saída probabilística.
- Cinco sementes na maioria das comparações; dez na varredura do *survival bonus* na regra original e na decomposição
  com N = 8c.
- A avaliação off-policy estima o valor por passo, sem a duração da sessão e sem a máscara de itens já vistos.

## Estado

Fase experimental encerrada em 17/11/2026. As últimas execuções (sementes 5–9 com N = 8 e a janela corrigida) e os
logs que estavam só na 4090 entraram no repositório em 18/11/2026; nessa data, `python verifica_resultados.py`
termina com `TUDO CONFERE`. Versão congelada dos experimentos: tag `v1.0-experimentos`.
