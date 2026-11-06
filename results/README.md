# results/: o que cada arquivo é

Tudo aqui veio da RTX 4090 (`exporta_resultados.sh`) ou das análises em CPU no Mac. Nada grande entra no
repositório: o registro bruto do P_E e as matrizes do modelo de usuário entram só pelo SHA-256.

| Caminho | Conteúdo | Gerado por |
|---|---|---|
| `KuaiRec.zip.sha256`, `kuairec_validation.json` | Hash dos dados brutos; conferência contra a Tabela 1 | README, seção 3 |
| `timing_4.2.txt` | Tempo de 2 épocas do MBPO, usado para estimar as execuções longas | README, seção 4.2 |
| `ambiente/` | Commit do DORL-codes, diff local (só o `pe_log.patch`), `uv pip freeze` do venv, hash das matrizes | `exporta_resultados.sh` |
| `logs/` | Logs de todas as execuções, inclusive as que não entram em nenhum resultado (abaixo) | `exporta_resultados.sh` |
| `table2/` | Um resumo por log no formato da Tabela 2, nos três protocolos | `table2_from_log.py --all-protocols` |
| `analises/pe_diag.json` | S1: diagnóstico do P_E sem treino | `diagnostico_pe.py` |
| `analises/rhat_aval.json` | S2: r_hat contra a recompensa de avaliação | `r_hat_vs_avaliacao.py` |
| `e1/aleatoria.json` | S3: política aleatória, 5 sementes, três protocolos | `politica_aleatoria.py` |
| `e1/e1_*.csv`, `e1/e1_saida.txt` | E1: valores por semente, resumo entre sementes, posições, inversões | `e1_protocolos.py` |
| `pe/pe_log_resumo.txt`, `pe/pe_log_DORL_s2.sha256` | P: resumo do P_E registrado no treino; hash do CSV bruto (fica na 4090) | `analisa_pe_log.py` |
| `pe/comandos_P_E2.md`, `pe/verifica_E2.txt` | Comandos do P e do E2; prova de que o E2 usou `PE_CONST=1.9609` | `verifica_E2.sh` |
| `versions_INVALIDO.txt` | Histórico apenas: o `pip freeze` antigo pegou o Python do sistema | `setup.sh` antigo |

## Logs que entram nos resultados

5 sementes de MBPO, MOPO, DORL e DORL-const (200 épocas cada), mais UCB e ε-greedy (1 execução cada).
Quando há mais de um log da mesma semente, vale o mais recente com 200 épocas; o `e1_protocolos.py` aplica
essa regra e lista o que usou no começo de `e1/e1_saida.txt`.

## Logs que não entram em nenhum resultado

| Log | Motivo |
|---|---|
| `[DORL_s0]_2026_10_22-12_34_16.log` | Completo, mas repetido; vale o de 29/10, 02:05 |
| `[DORL_s0]_2026_10_29-01_31_01.log` | Interrompido (95 épocas) |
| `[DORL_s0]_2026_10_29-01_47_08.log` | Interrompido (68 épocas) |
| `[DORL_s2]_2026_10_22-13_36_51.log` | Interrompido (135 épocas) |
| `[DORL_pelog_s2]_*.log` | Execução do P; métricas idênticas às do `DORL_s2` (o patch não altera a recompensa) |
| `[MBPO-timing]_*.log` | Cronometragem de 2 épocas (`timing_4.2.txt`) |
| `[pointneg]_*.log` | Treino do modelo de usuário (DeepFM em ensemble), não é uma política |
