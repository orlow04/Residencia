# Passo 1 — Pipeline do DORL com versões congeladas (KuaiRec)

Objetivo deste passo: sair com (1) ambiente reprodutível, (2) dados do KuaiRec validados contra a
Tabela 1 e (3) a primeira métrica da Tabela 2 (MBPO no KuaiRec), mais os baselines UCB e ε-greedy.

| Arquivo | Para que serve |
|---|---|
| `setup.sh` | Clona o DORL-codes (commit `818c9bd`) e o fork do tianshou (`0f59e38`), cria o ambiente Python 3.10 e grava `versions.txt` |
| `requirements.txt` | Versões fixadas e o motivo de cada uma |
| `smoke_test.sh` + `make_synthetic_kuairec.py` | Roda a pipeline inteira com dados sintéticos em ~2 min, sem baixar nada |
| `validate_kuairec.py` | Confere usuários, itens, interações e categorias contra a Tabela 1 e grava os SHA-256 dos dados |
| `table2_from_log.py` | Resume um log no formato da Tabela 2, com a mesma agregação do script dos autores |

O que já foi verificado (20/10/2026, CPU, sem GPU):
- `setup.sh` + `smoke_test.sh` rodam de ponta a ponta numa pasta limpa: pré-processamento,
  modelo de usuário, DORL, UCB e ε-greedy.
- `table2_from_log.py` aplicado aos **logs originais dos autores** (`results_for_paper/results_all`)
  reproduz **exatamente** os 40 valores da Tabela 2 (5 métodos × 4 métricas × 2 datasets).

O que **não** foi verificado: execução com os dados reais e em GPU.

---

## 1. Ambiente

Rode todos os comandos a partir desta pasta (`dorl_passo1/`).

```bash
bash setup.sh dorl-workspace
source dorl-workspace/.venv/bin/activate
```

No fim, o script testa a GPU. O `torch==1.13.1` do PyPI vem com CUDA 11.7 e exige driver NVIDIA ≥ 515.
**Se a GPU for de geração recente** (RTX 40xx, H100 ou RTX 50xx), confira a mensagem do teste: o
torch 1.13 pode não ter suporte a ela. Nesse caso, me diga o modelo da GPU antes de trocar versões.

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
  for f in saved_models/KuaiEnv-v0/A2C_with_emb/logs/\[${m}_s*\]_*.log; do
    python ../../table2_from_log.py "$f" --dataset KuaiRec --method $m --all-protocols
  done
done
```

O que esperar no KuaiRec (Tabela 2): o ganho do DORL sobre o MBPO está em **Length** (26.7 contra 15.6)
e, por consequência, em **R_tra** (20.5 contra 12.0). R_each e MCD ficam praticamente iguais aos do MBPO
(MCD 0.379 contra 0.362). Se a média entre sementes do DORL não ficar claramente acima do MBPO em
Length e R_tra, a afirmação principal do artigo não se reproduziu aqui.

## Registro para o apêndice técnico

Guarde: `versions.txt`, `KuaiRec.zip.sha256`, `kuairec_validation.json`, os logs de `saved_models/`,
o tempo medido em 4.2 e a saída do `table2_from_log.py`.
