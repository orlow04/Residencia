# results/: o que cada arquivo é

Tudo aqui veio da RTX 4090 (pelo `exporta_resultados.sh` ou pela opção `--out` das análises) ou das análises em CPU
no Mac. Nada grande entra no repositório: trajetórias, registro bruto do P_E, matrizes do modelo de usuário e
checkpoints entram só pelo SHA-256. Para conferir que as tabelas vêm dos logs, rode `python verifica_resultados.py`
na raiz.

| Caminho | Conteúdo | Gerado por | Registro |
|---|---|---|---|
| `KuaiRec.zip.sha256`, `kuairec_validation.json` | Hash dos dados brutos; conferência contra a Tabela 1 do artigo | `validate_kuairec.py` | `docs/reproducao.md`, seção 3 |
| `timing_4.2.txt` | Tempo de 2 épocas do MBPO, usado para estimar as execuções longas | `docs/reproducao.md`, seção 4.2 | — |
| `ambiente/` | Commit do DORL-codes, diff dos patches aplicados na 4090, versões do venv e hash das matrizes do modelo de usuário | `exporta_resultados.sh` | `docs/reproducao.md`, seção 11 |
| `logs/` | Logs de todas as execuções (políticas, bandits, modelos de usuário), inclusive as descartadas (abaixo) | `exporta_resultados.sh` | — |
| `trajetorias/` | SHA-256 de cada CSV de trajetória conferido (`*.csv.ok`; os CSVs ficam na 4090) e a conferência do `traj_log.patch` (`verifica_traj.txt`) | `roda_*.sh`, `verifica_traj.sh` | `docs/semana9.md`, etapa 2 |
| `table2/` | Um resumo por log no formato da Tabela 2 (semana 8) | `table2_from_log.py` | `docs/experimentos_semana8.md` |
| `analises/` | S1 (`pe_diag.json`) e S2 (`rhat_aval.json`) | `diagnostico_pe.py`, `r_hat_vs_avaliacao.py` | idem |
| `e1/` | S3 (`aleatoria.json`) e os resumos entre sementes de R, E1, E2 e bandits (`e1_*`) | `politica_aleatoria.py`, `e1_protocolos.py` | idem; `docs/semana10.md`, etapa 3 |
| `pe/` | P: resumo do P_E no treino e hash do registro bruto; comandos do P e do E2; prova de que o E2 usou `PE_CONST=1.9609` | `analisa_pe_log.py`, `verifica_E2.sh` | `docs/experimentos_semana8.md` |
| `bolha/` | Variedade por trajetória contra as janelas do próprio usuário | `bolha.py` | `docs/semana9.md`, etapa 3 |
| `e5/` | E5: resumos entre sementes, modelo de usuário `rawwr` (resumo e hashes) e S2 com esse modelo | `roda_e5.sh`, `e1_protocolos.py`, `r_hat_vs_avaliacao.py` | etapa 4 |
| `e3/` | E3: resumos com N = 1 e N = 8; política aleatória com N = 1, 4c, 8 e 8c | `e1_protocolos.py`, `politica_aleatoria.py` | etapa 5 |
| `a5/` | Janela corrigida (N = 4c e 8c) para os quatro métodos | `e1_protocolos.py` | etapa 6; `docs/semana10.md`, etapa 2 |
| `fb/`, `mecanismo/` | Estrutura das trajetórias FB (ciclos); regra de saída contrafactual | `fb_rotacao.py`, `mecanismo_bolha.py` | `docs/semana9.md`, etapa 7 |
| `alem/` | Cobertura, Gini, novidade, serendipidade e personalização de todas as variantes (inclusive `MOPO-tmax`) | `alem_da_acuracia.py` | etapa 8; `docs/semana10.md`, etapa 4 |
| `sessoes/` | Regra de saída contra as sessões reais do big_matrix | `sessoes_reais.py` | `docs/semana9.md`, etapa 9 |
| `e6/` | E6: resumos entre sementes; política aleatória com a saída calibrada | `e1_protocolos.py`, `politica_aleatoria.py` | etapa 10 |
| `revisitas/` | Revisitas reais (sem duplicatas) e repetição das políticas no FB | `revisitas.py` | `docs/semana10.md`, etapa 1 |
| `sb/` | Varredura do *survival bonus*, regras N4 e E6, sementes 0–4 | `e1_protocolos.py` | etapa 4 |
| `tmax/` | `MOPO-tmax` (MOPO com o `train_max` do DORL), regras N4 e E6, sementes 0–4 | `e1_protocolos.py` | etapa 4 |
| `s10/` | 10 sementes na regra N4 (`n4_*`) e com N = 8c (`n8c_*`), controle inverso no E6 (`e6_*`) e critérios além do artigo na regra N4 e no E6 (`alem/`) | `e1_protocolos.py`, `alem_da_acuracia.py` | etapas 4 e 5 |
| `ope/` | Avaliação off-policy: validações semi-sintéticas, log real, políticas e comparação com o simulador | `ope.py`, `ope_contexto.py`, `ope_dados.py`, `ope_politicas.py`, `ope_compara.py` | `docs/off_policy.md` |

## Logs que entram nos resultados

Cada resumo do `e1_protocolos.py` lista, no começo da saída de texto (`*_saida.txt`), os logs usados e os
descartados. Quando há mais de um log completo da mesma mensagem, vale o mais recente; logs com menos de 200 épocas
são descartados. O `verifica_resultados.py` reconstrói os globs de cada resumo a partir dos `*_por_semente.csv`.

## Logs que não entram em nenhum resultado

| Log | Motivo |
|---|---|
| `[DORL_s0]_2026_10_22-12_34_16.log` | Completo, mas repetido; vale o de 29/10, 02:05 |
| `[DORL_s0]_2026_10_29-01_31_01.log` | Interrompido (95 épocas) |
| `[DORL_s0]_2026_10_29-01_47_08.log` | Interrompido (68 épocas) |
| `[DORL_s2]_2026_10_22-13_36_51.log` | Interrompido (135 épocas) |
| `[DORL_pelog_s2]_*.log` | Execução do P; métricas idênticas às do `DORL_s2` (o patch não altera a recompensa) |
| `[E2check_s*]_*.log`, `[E2control_s0]_*.log` | Conferência do E2: 2 épocas por semente (`verifica_E2.sh`) |
| `[TRAJcheck_s2]_*.log` | Conferência do `traj_log.patch`: 2 épocas (`verifica_traj.sh`) |
| `[MBPO-timing]_*.log` | Cronometragem de 2 épocas (`timing_4.2.txt`) |
| `[pointneg]_*.log`, `[rawwr]_*.log` | Treino dos modelos de usuário (original e do E5), não são políticas |
| `[MBPO_traj_s*]_*.log`, `[DORL_traj_s*]_*.log` | Reexecuções das sementes 0–4 com registro das trajetórias (`roda_trajetorias.sh`); idênticas às originais. As de MOPO e DORL-const entram em `s10/n4_*` |
| `[CK_*_s*]_*.log` | Retreino com checkpoint, fonte das políticas da avaliação off-policy (`roda_ckpt.sh`); idêntico às execuções originais |

As reexecuções das duas últimas linhas (e as de MOPO e DORL-const, sementes 0–4) são conferidas valor por valor
contra as execuções originais de `e1/` pelo `verifica_resultados.py`.
