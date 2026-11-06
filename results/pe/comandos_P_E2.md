# Comandos do P (registro do P_E) e do E2 (DORL com P_E constante)

Rodados na RTX 4090, em `dorl-workspace/DORL-codes`, com o `pe_log.patch` aplicado
(prova: `results/ambiente/dorl_codes_diff.patch`). Os argumentos abaixo são os gravados no cabeçalho dos logs;
o histórico do shell se perdeu num reinício da máquina, e as variáveis de ambiente foram confirmadas como descrito no fim.

```bash
ARGS="--env KuaiEnv-v0 --cuda 0 --num_leave_compute 4 --leave_threshold 0 --which_tracker avg \
  --reward_handle cat --window_size 3 --read_message pointneg --lambda_variance 0.05 --lambda_entropy 5"

# P: DORL semente 2 com registro do P_E (log: [DORL_pelog_s2]_2026_10_29-03_26_07.log)
PE_LOG=pe_log_DORL_s2.csv python run_Policy_Main.py $ARGS --seed 2 --message DORL_pelog_s2
python ../../analisa_pe_log.py pe_log_DORL_s2.csv      # -> results/pe/pe_log_resumo.txt; c = 1.9609

# E2: DORL com P_E constante, 5 sementes (logs: [DORLconst_s0..4]_2026_10_29-*.log)
for s in 0 1 2 3 4; do
  PE_CONST=1.9609 python run_Policy_Main.py $ARGS --seed $s --message DORLconst_s$s
done
```

O CSV bruto do P (20.230.715 passos) ficou na 4090; o hash está em `pe_log_DORL_s2.sha256`.

## Verificação da constante (06/11/2026)

`verifica_E2.sh` repetiu 2 épocas de cada semente com `PE_CONST=1.9609`: as épocas 1 e 2 são idênticas às dos
logs originais nas 5 sementes, e o controle sem `PE_CONST` difere (`verifica_E2.txt`). O treino é determinístico
(`[MBPO-timing]` com 2 épocas = 2 primeiras épocas do `[MBPO_s2]`), então isso confirma a constante usada.
Os logs `E2check_s*` e `E2control_s0` dessa verificação ficaram na 4090 e não entram em nenhum resultado.
