# Reprodução passo a passo

Guia para refazer os experimentos do zero. Para conferir os resultados sem refazer nada, veja o `README.md` (seção
"Como conferir"). Todos os comandos rodam a partir da raiz do repositório, exceto quando indicado
`dorl-workspace/DORL-codes`. Os scripts daqui são chamados de lá como `../../script.py`. Cada script tem, no
cabeçalho, o uso e as opções.

As seções 1 a 10 cobrem a semana 8 (reprodução, S1–S3, P, E2, E1). As seções 11 e 12 cobrem as semanas 9 e 10, e a
seção 13, a verificação.

## 1. Ambiente

Rode todos os comandos a partir desta pasta (`dorl_passo1/`).

```bash
bash setup.sh dorl-workspace
source dorl-workspace/.venv/bin/activate
```

No fim, o script testa a GPU. O `torch==1.13.1` do PyPI vem com CUDA 11.7 e exige driver NVIDIA ≥ 515. Todas as
execuções deste trabalho usaram essa versão numa RTX 4090 (versões exatas em `results/ambiente/versions_venv.txt`).
Em outra GPU, confira a mensagem do teste antes de trocar versões. No Mac (só análises em CPU), use
`requirements-macos.txt`.

## 2. Teste de fumaça (antes dos dados reais)

```bash
bash smoke_test.sh dorl-workspace
```

Deve terminar com `TESTE DE FUMAÇA CONCLUÍDO`. O script apaga os dados sintéticos e os caches no
final; os ambientes gravam caches na pasta de dados e, se ficassem, contaminariam a execução real.

## 3. Dados reais do KuaiRec (fonte oficial, sem depender do servidor dos autores)

```bash
cd dorl-workspace
wget https://zenodo.org/records/18164998/files/KuaiRec.zip
sha256sum KuaiRec.zip > KuaiRec.zip.sha256
unzip KuaiRec.zip -d kuairec_raw
mkdir -p DORL-codes/environments/KuaiRec/data
for f in big_matrix.csv small_matrix.csv user_features.csv item_categories.csv; do
  cp "$(find kuairec_raw -name "$f" | head -1)" DORL-codes/environments/KuaiRec/data/
done

cd DORL-codes
python preprocessing_kuairec.py
python ../../validate_kuairec.py --data environments/KuaiRec/data --out ../kuairec_validation.json
```

**Critério para seguir:** a validação precisa terminar com `TODOS OS CAMPOS CONFEREM`. Se algum campo
divergir, pare e investigue; a divergência é um achado, não um detalhe.

## 4. Primeira reprodução: MBPO, UCB e ε-greedy no KuaiRec

As configurações abaixo foram tiradas dos **logs originais** dos autores, não do README do repositório.
Há uma diferença importante: o README usa `--num_leave_compute 1` para as políticas de RL, mas os logs
que geraram a Tabela 2 usam **`4`** para todos os métodos (que corresponde ao N = 4 do artigo). Os logs
também mostram a semente usada: **2** para MBPO, MOPO e DORL no KuaiRec, e **0** para os bandits.

```bash
# 4.1 Modelo de usuário (DeepFM em ensemble), comando do README
python run_worldModel_ensemble.py --env KuaiEnv-v0 --cuda 0 --epoch 5 --loss pointneg --message pointneg

# 4.2 Cronometrar ANTES da execução longa: 2 épocas
time python run_Policy_Main.py --env KuaiEnv-v0 --seed 2 --cuda 0 --num_leave_compute 4 --leave_threshold 0 \
  --which_tracker avg --reward_handle cat --lambda_variance 0 --lambda_entropy 0 --window_size 3 \
  --read_message pointneg --message MBPO-timing --epoch 2

# 4.3 Execução completa (200 épocas), em segundo plano
nohup python run_Policy_Main.py --env KuaiEnv-v0 --seed 2 --cuda 0 --num_leave_compute 4 --leave_threshold 0 \
  --which_tracker avg --reward_handle cat --lambda_variance 0 --lambda_entropy 0 --window_size 3 \
  --read_message pointneg --message MBPO_s2 > mbpo_s2.out 2>&1 &

# 4.4 Baselines (independentes do modelo de usuário acima; podem rodar em paralelo)
nohup python run_linUCB.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 \
  --cuda 0 --loss pointneg --message UCB > ucb.out 2>&1 &
nohup python run_epsilongreedy.py --env KuaiEnv-v0 --num_leave_compute 4 --leave_threshold 0 --epoch 200 --seed 0 \
  --cuda 0 --loss pointneg --message epsilon-greedy > eps.out 2>&1 &
```

**Não altere** `--training-num`, `--test-num` nem `--episode-per-collect`. Valores diferentes dos
padrões (100) quebram o coletor do tianshou com `AssertionError`.

## 5. Comparar com a Tabela 2

```bash
python ../../table2_from_log.py "saved_models/KuaiEnv-v0/A2C_with_emb/logs/[MBPO_s2]_"*.log \
  --dataset KuaiRec --method MBPO --all-protocols
python ../../table2_from_log.py "saved_models/KuaiEnv-v0/LinUCB/logs/[UCB]_"*.log --dataset KuaiRec --method UCB
python ../../table2_from_log.py "saved_models/KuaiEnv-v0/EpsilonGreedy/logs/[epsilon-greedy]_"*.log \
  --dataset KuaiRec --method epsilon-greedy
```

Como ler a saída:
- O `±` do artigo é a dispersão **entre épocas de uma execução**, não entre sementes. "Dentro de um
  desvio" é um critério frouxo; registre também a diferença bruta.
- Os nomes dos logs originais terminam em `_s2` e `_s4`, o que sugere que os autores rodaram várias
  sementes e reportaram uma. Isso é hipótese, não fato verificado. Por isso, depois da semente 2,
  rode as sementes 0, 1, 3 e 4 e reporte a média entre sementes.
- `--all-protocols` imprime também FB e NX_10. Não são da Tabela 2, mas são os dados do experimento
  de inversão por protocolo; guarde todos os logs em `saved_models/`.

## 6. DORL (e MOPO) no KuaiRec

Configuração tirada do log original dos autores (`results_for_paper/results_all/kuairec/[DORL]_*.log`).
O DORL usa os **mesmos** argumentos do MBPO; mudam só as duas penalidades:

| Método | `--lambda_variance` | `--lambda_entropy` |
|---|---|---|
| MBPO | 0 | 0 |
| MOPO | 0.05 | 0 |
| DORL | 0.05 | 5 |

**Atenção:** o primeiro comando de DORL no README do repositório usa `--lambda_variance 0.01 --lambda_entropy 0.05`.
Não é o que gerou a Tabela 2. O log dos autores e a lista de comandos mais abaixo no mesmo README usam `0.05` e `5`.

O modelo de usuário é o mesmo da seção 4.1 (`--read_message pointneg`); não precisa treinar de novo.
No log dos autores, o DORL levou ~14% mais tempo que o MBPO; aqui, conte ~45–50 min por semente.

```bash
# DORL, 5 sementes em sequência (rode dentro de tmux ou com nohup bash -c '...' &)
for s in 0 1 2 3 4; do
  python run_Policy_Main.py --env KuaiEnv-v0 --seed $s --cuda 0 --num_leave_compute 4 --leave_threshold 0 \
    --which_tracker avg --reward_handle cat --lambda_variance 0.05 --lambda_entropy 5 --window_size 3 \
    --read_message pointneg --message DORL_s$s > dorl_s$s.out 2>&1
done

# Opcional: MOPO (só a penalidade de variância). Isola o efeito da penalidade de entropia,
# que é a contribuição do DORL. Pode rodar em paralelo com o laço acima.
for s in 0 1 2 3 4; do
  python run_Policy_Main.py --env KuaiEnv-v0 --seed $s --cuda 0 --num_leave_compute 4 --leave_threshold 0 \
    --which_tracker avg --reward_handle cat --lambda_variance 0.05 --lambda_entropy 0 --window_size 3 \
    --read_message pointneg --message MOPO_s$s > mopo_s$s.out 2>&1
done
```

Comparar com a Tabela 2:

```bash
for m in DORL MOPO; do
  for f in saved_models/KuaiEnv-v0/A2C_with_emb/logs/[[]${m}_s[0-9][]]_*.log; do   # filtro estrito
    python ../../table2_from_log.py "$f" --dataset KuaiRec --method $m --all-protocols
  done
done
```

O que esperar no KuaiRec (Tabela 2): o ganho do DORL sobre o MBPO está em **Length** (26.7 contra 15.6)
e, por consequência, em **R_tra** (20.5 contra 12.0). R_each e MCD ficam praticamente iguais aos do MBPO
(MCD 0.379 contra 0.362). Se a média entre sementes do DORL não ficar claramente acima do MBPO em
Length e R_tra, a afirmação principal do artigo não se reproduziu aqui.

Se uma semente foi interrompida e repetida, há mais de um log para ela. O `e1_protocolos.py` (seção 9)
descarta os incompletos e os repetidos e lista quais usou; o laço acima não faz isso.

## 7. Análises sem treino (S1, S2, S3), em CPU

Rodam no Mac, de `dorl-workspace/DORL-codes`, com os dados da seção 3. S1 e S2 precisam das matrizes do
modelo de usuário (`find saved_models -name "*_mat*.pickle"`; hash em `results/ambiente/mats.sha256`).

```bash
# S1: quanto do P_E é sinal e quanto é preenchimento
python ../../diagnostico_pe.py --data environments/KuaiRec/data \
  --matpre matPre.pickle --matvar matVar.pickle --out pe_diag.json
# S2: o r_hat prevê a recompensa usada na avaliação?
python ../../r_hat_vs_avaliacao.py --data environments/KuaiRec/data --matpre matPre.pickle --out rhat_aval.json
# S3: política aleatória nos três protocolos (referência do E1)
python ../../politica_aleatoria.py --data environments/KuaiRec/data \
  --domination environments/KuaiRec/env/feature_domination.pickle --seeds 0 1 2 3 4 --out aleatoria.json
```

Saídas: `results/analises/pe_diag.json`, `results/analises/rhat_aval.json`, `results/e1/aleatoria.json`.

## 8. Registro do P_E (P) e DORL com P_E constante (E2), na 4090

O `pe_log.patch` altera só o `simulated_env.py`. Sem `PE_LOG` e `PE_CONST`, a recompensa é idêntica à
original (`tests/test_patch.py`).

```bash
git apply ../../pe_log.patch        # em dorl-workspace/DORL-codes
```

Os comandos exatos estão em `results/pe/comandos_P_E2.md`. Em resumo: o DORL da semente 2 roda com
`PE_LOG=pe_log_DORL_s2.csv`, o `analisa_pe_log.py` dá a constante (média do P_E na última fase,
c = 1,9609), e as cinco sementes `DORLconst_s*` rodam com `PE_CONST=1.9609`.

O `verifica_E2.sh` confirma a constante sem depender do histórico do shell. Como o treino é
determinístico, ele repete 2 épocas por semente e compara com os logs (`results/pe/verifica_E2.txt`).
Use um arquivo novo de `PE_LOG` a cada execução: o patch abre o arquivo em modo de acréscimo.

## 9. E1: inversão por protocolo

```bash
L=saved_models/KuaiEnv-v0
python ../../e1_protocolos.py \
  "MBPO=$L/A2C_with_emb/logs/[[]MBPO_s[0-9][]]_*.log" \
  "MOPO=$L/A2C_with_emb/logs/[[]MOPO_s[0-9][]]_*.log" \
  "DORL=$L/A2C_with_emb/logs/[[]DORL_s[0-9][]]_*.log" \
  "DORL-const=$L/A2C_with_emb/logs/[[]DORLconst_s[0-9][]]_*.log" \
  "UCB=$L/LinUCB/logs/[[]UCB[]_]*.log" \
  "eps-greedy=$L/EpsilonGreedy/logs/[[]epsilon-greedy[]_]*.log" \
  --aleatoria aleatoria.json --out e1 > e1_saida.txt
```

O glob `[[]UCB[]_]*` pega a semente 0 (`[UCB]_…`) e as sementes 1–4 (`[UCB_s1]_…`, `roda_bandits.sh`). Para refazer
todas as tabelas a partir de `results/logs/`, sem a 4090, use o `verifica_resultados.py` (seção 13).

## 10. Levar os resultados para o repositório

Na 4090, na raiz do repositório e com o venv ativo, rode `bash exporta_resultados.sh`. Ele copia os logs de todas as
execuções e os marcadores das trajetórias (SHA-256 de cada CSV conferido), refaz o resumo do P_E e grava o commit e o
diff do DORL-codes, as versões do venv e os hashes das matrizes. Depois, em qualquer máquina, `python
verifica_resultados.py` confere as tabelas (seção 13) e `python figuras.py` refaz `docs/figuras/`. O índice de
`results/` está em `results/README.md`.

## 11. Patches no DORL-codes (semanas 9 e 10)

Os experimentos das semanas 9 e 10 alteram o DORL-codes por patches. Cada patch só muda o comportamento quando a sua
variável de ambiente está definida; sem ela, o código se comporta como o original (provado em `tests/`). Na 4090,
os seis ficaram aplicados juntos, nesta ordem (testado com `git apply` sobre o commit `818c9bd`):

```bash
cd dorl-workspace/DORL-codes
for p in pe_log traj_log e5_alvo a5_janela e6_saida_prob tmax; do git apply ../../$p.patch; done
```

| Patch | Variável | Efeito |
|---|---|---|
| `pe_log.patch` | `PE_LOG`, `PE_CONST` | Registra o P_E por passo de treino; com `PE_CONST`, troca o P_E por uma constante (DORL-const) |
| `traj_log.patch` | `TRAJ_LOG` | Grava as trajetórias de avaliação (usuário, item, recompensa, categoria por passo) |
| `e5_alvo.patch` | `E5_RAW_WR` | Modelo de usuário treinado com o watch_ratio bruto limitado a 5 (alvo da avaliação) |
| `a5_janela.patch` | `A5_FIX` | Janela da regra de saída sem o recorte nos primeiros passos |
| `e6_saida_prob.patch` | `E6_P_REP`, `E6_P_BASE` | Saída probabilística calibrada nas sessões reais |
| `tmax.patch` | `TMAX_LAMBDA` | Usa o `train_max` (escala da recompensa no estado) de outro `lambda_entropy` |

O diff efetivamente aplicado na 4090 está em `results/ambiente/dorl_codes_diff.patch`.

## 12. Execuções e análises das semanas 9 e 10

As execuções rodam na 4090, na raiz do repositório, com o venv ativo. Os scripts de execução conferem as 200 épocas
de cada execução e pulam o que já terminou; a maioria confere também se os patches necessários estão aplicados. As análises rodam de `dorl-workspace/DORL-codes`
(onde estão os dados e as trajetórias) ou da raiz, como indicado no cabeçalho de cada script.

| Experimento | Execução (4090) | Análise | Saída |
|---|---|---|---|
| Trajetórias | `bash verifica_traj.sh`; `roda_trajetorias.sh MBPO DORL` e `MOPO DORL-const` | — | `results/trajetorias/` (SHA-256) |
| Bolha | — | `bolha.py --traj trajetorias` | `results/bolha/` |
| E5 (alvo da avaliação) | `roda_e5.sh modelo`, depois `roda_e5.sh MBPO` e `DORL` | `e1_protocolos.py`; `r_hat_vs_avaliacao.py` com as matrizes `rawwr` | `results/e5/` |
| E3 (N da regra) | `roda_e3.sh 1` e `roda_e3.sh 8` | `e1_protocolos.py`; `politica_aleatoria.py --num_leave_compute N` | `results/e3/` |
| A5 (janela corrigida) | `roda_a5.sh 4` e `8`; `METODOS="MOPO DORLconst" roda_a5.sh 4` e `8` | `e1_protocolos.py`; `politica_aleatoria.py --janela_correta` | `results/a5/` |
| Mecanismo e FB | — | `mecanismo_bolha.py`, `fb_rotacao.py` | `results/mecanismo/`, `results/fb/` |
| Critérios além do artigo | — | `alem_da_acuracia.py` | `results/alem/` |
| Sessões reais | — | `sessoes_reais.py` (raiz, Mac) | `results/sessoes/` |
| E6 (saída calibrada) | `roda_e6.sh MBPO`, `MOPO`, `DORL`, `DORLconst` | `e1_protocolos.py`; `politica_aleatoria.py --p_saida 0.063 0.026 --num_leave_compute 1` | `results/e6/` |
| Revisitas | — | `revisitas.py` (partes A e B no Mac; C com `--traj`) | `results/revisitas/` |
| Bandits (sementes 1–4) | `SEEDS="1 2" roda_bandits.sh UCB` e `EG` (e `SEEDS="3 4"`) | `e1_protocolos.py` | `results/e1/` |
| *Survival bonus* | `BONUS="0 0.5 1 2 4" roda_bonus.sh N4` e `E6` | `e1_protocolos.py`; `alem_da_acuracia.py` | `results/sb/` |
| `train_max` | `roda_tmax.sh N4` e `E6`; `METODOS="MOPO SB0"` para os dois sentidos | `e1_protocolos.py`; `alem_da_acuracia.py` | `results/tmax/`, `results/alem/` |
| Sementes 5–9 (regra N4) | `SEEDS="5 6 7 8 9"` em `roda_bonus.sh N4`, `roda_tmax.sh N4` e `roda_trajetorias.sh MOPO DORL-const` | `e1_protocolos.py`; `alem_da_acuracia.py` | `results/s10/` |
| Sementes 5–9 (N = 8c) | `SEEDS="5 6 7 8 9"` em `roda_a5.sh 8` e `METODOS="MOPO DORLconst" roda_a5.sh 8` | `e1_protocolos.py` (com `results/e3/aleatoria_N8c.json`) | `results/s10/n8c_*` |
| OPE | `roda_ckpt.sh` (checkpoints da época 200) | `ope.py --validacao`, `ope_contexto.py`, `ope_dados.py`, `ope_politicas.py`, `ope_compara.py` | `results/ope/` |

Os globs do `e1_protocolos.py` de cada conjunto são os que o `verifica_resultados.py` reconstrói a partir dos
`*_por_semente.csv`; para refazer um resumo, o caminho mais curto é copiar o comando que ele monta. Os comandos
exatos de cada etapa, com as ressalvas, estão nos registros (`docs/semana9.md`, `docs/semana10.md`,
`docs/off_policy.md`).

## 13. Verificação

```bash
python verifica_resultados.py      # recalcula as tabelas a partir de results/logs (CPU, ~20 s)
bash tests/roda_testes.sh          # testes de fidelidade dos scripts e dos patches (CPU, ~1 min; exige o dorl-workspace)
```

O primeiro confere os resumos entre sementes (`results/{e1,a5,e3,e5,e6,sb,tmax,s10}`), os resumos por log
(`results/table2`), a comparação OPE × simulador e as reexecuções: as 20 com registro das trajetórias e as 20 com
checkpoint (as políticas do OPE) têm de reproduzir, valor por valor, as execuções originais de `results/e1`. Um
conjunto cujos logs não estejam em `results/logs` aparece como INCOMPLETO.
