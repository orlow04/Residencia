#!/usr/bin/env bash
# Roda todos os testes de fidelidade descritos em tests/README.md e imprime um resumo.
# Uso (da raiz do repositório, no Mac ou na 4090):  bash tests/roda_testes.sh
# Precisa do dorl-workspace criado pelo setup.sh: usa o venv, os arquivos originais do DORL-codes (lidos do git, então
# o estado dos patches no workspace não importa) e, só no test_e5, os dados reais do KuaiRec. Nada usa GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
W="$R/dorl-workspace/DORL-codes"
PY="${PY:-$R/dorl-workspace/.venv/bin/python}"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
[ -d "$W/.git" ] || { echo "falta $W (rode o setup.sh)"; exit 1; }

orig() { git -C "$W" show "HEAD:$1" > "$2"; }          # arquivo original do DORL-codes (commit 818c9bd)
com_patch() { cp "$1" "$2" && patch -s "$2" "$3"; }      # cópia do original com um patch deste repositório
passou=0; falhou=()
roda() {  # roda NOME DIRETÓRIO comando...: mostra a última linha; em caso de falha, a saída inteira
  local nome="$1" dir="$2"; shift 2
  if (cd "$dir" && "$@") > "$T/$nome.out" 2>&1; then
    local msg; msg="$(grep '^OK' "$T/$nome.out" | tail -1)"; [ -n "$msg" ] || msg="$(tail -1 "$T/$nome.out")"
    printf "  %-28s %s\n" "$nome" "$(echo "$msg" | cut -c1-90)"; passou=$((passou + 1))
  else
    printf "  %-28s FALHOU\n" "$nome"; sed 's/^/      /' "$T/$nome.out" | tail -15; falhou+=("$nome")
  fi
}

echo "Dados sintéticos em $T"
"$PY" make_synthetic_kuairec.py --out "$T/synth" --n_users 500 --n_items 120 --small_items 100 --small_users 50 \
  --per_user 80 > /dev/null && "$PY" tests/prepara_dados_teste.py "$T/synth" "$T/rtest" > /dev/null || exit 1
orig src/core/worldModel/simulated_env.py "$T/simenv_orig.py"
orig src/core/evaluation/evaluator.py "$T/eval_orig.py"
orig environments/KuaiRec/env/KuaiEnv.py "$T/kuai_orig.py"
orig run_Policy_Main.py "$T/rpm_orig.py"
com_patch "$T/simenv_orig.py" "$T/simenv_pe.py" pe_log.patch
com_patch "$T/eval_orig.py" "$T/eval_traj.py" traj_log.patch
com_patch "$T/kuai_orig.py" "$T/kuai_a5.py" a5_janela.patch
com_patch "$T/kuai_orig.py" "$T/kuai_e6.py" e6_saida_prob.patch
com_patch "$T/kuai_orig.py" "$T/kuai_e5.py" e5_alvo.patch
com_patch "$T/kuai_e5.py" "$T/kuai_tres.py" a5_janela.patch && patch -s "$T/kuai_tres.py" e6_saida_prob.patch
com_patch "$T/rpm_orig.py" "$T/rpm_tmax.py" tmax.patch

echo "Scripts de análise"
roda test_ref "$R" "$PY" tests/test_ref.py "$R/diagnostico_pe.py" "$T/synth"
roda test_evmat "$R" "$PY" tests/test_evmat.py "$R/r_hat_vs_avaliacao.py" "$T/rtest"
roda test_random "$R" "$PY" tests/test_random.py "$R/politica_aleatoria.py" "$T/rtest" "$T/rtest/dom.pickle"
roda test_bolha "$R" "$PY" tests/test_bolha.py "$R/bolha.py"
roda test_alem "$R" "$PY" tests/test_alem.py "$R/alem_da_acuracia.py"
roda test_mecanismo "$R" "$PY" tests/test_mecanismo.py "$R/mecanismo_bolha.py" "$R/fb_rotacao.py"
roda test_sessoes "$R" "$PY" tests/test_sessoes.py "$R/sessoes_reais.py"
roda test_ope "$R" "$PY" tests/test_ope.py "$R/ope.py"
roda test_ope_politicas "$R" "$PY" tests/test_ope_politicas.py "$R/ope_politicas.py" "$W"
echo "Patches (original do git × cópia com o patch)"
roda test_patch "$R" "$PY" tests/test_patch.py "$T/simenv_orig.py" "$T/simenv_pe.py" "$T/synth" "$T/pe_log_teste.csv"
roda test_traj "$R" "$PY" tests/test_traj.py "$T/eval_orig.py" "$T/eval_traj.py" "$W/src/tianshou" "$T/traj_teste.csv"
roda test_a5 "$W" "$PY" "$R/tests/test_a5.py" "$T/kuai_orig.py" "$T/kuai_a5.py"
roda test_e6 "$W" "$PY" "$R/tests/test_e6.py" "$T/kuai_orig.py" "$T/kuai_e6.py"
roda "test_e6 (E5+A5+E6 juntos)" "$W" "$PY" "$R/tests/test_e6.py" "$T/kuai_orig.py" "$T/kuai_tres.py"
roda test_tmax "$W" "$PY" "$R/tests/test_tmax.py" "$T/rpm_orig.py" "$T/rpm_tmax.py"
if [ -f "$W/environments/KuaiRec/data/big_matrix.csv" ]; then
  roda test_e5 "$W" "$PY" "$R/tests/test_e5.py" "$T/kuai_orig.py" "$T/kuai_e5.py" environments/KuaiRec/data
else
  echo "  test_e5                      pulado: exige os dados reais em $W/environments/KuaiRec/data"
fi

echo
if [ ${#falhou[@]} -eq 0 ]; then echo "TODOS OS TESTES PASSARAM ($passou)"; else echo "FALHARAM: ${falhou[*]}"; exit 1; fi
