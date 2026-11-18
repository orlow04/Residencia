"""Recalcula, a partir dos logs em results/logs, as tabelas do repositório e compara com os arquivos commitados.

Roda em CPU, em poucos minutos, sem GPU e sem os dados do KuaiRec. É a forma mais rápida de conferir que cada número
das tabelas vem dos logs das execuções.

O que é conferido:
  1. Resumos entre sementes do e1_protocolos.py (results/{e1,a5,e3,e5,e6,sb,tmax,s10}): os métodos e os logs de cada
     método são lidos do *_por_semente.csv commitado; os globs são refeitos a partir deles; o script roda de novo e
     os CSVs (por semente, resumo, posições) e a saída de texto precisam coincidir.
  2. Resumos por log no formato da Tabela 2 (results/table2), com o table2_from_log.py.
  3. Comparação OPE × simulador (results/ope/ope_comparacao.txt), com o ope_compara.py.
  4. Reexecuções das sementes 0–4 de MBPO, MOPO, DORL e DORL-const, com registro das trajetórias (*_traj_s*,
     roda_trajetorias.sh) e com checkpoint (CK_*_s*, roda_ckpt.sh; as políticas da avaliação off-policy): o treino é
     determinístico, então cada valor por semente precisa ser igual ao da execução original (results/e1).

Uso (da raiz do repositório):  python verifica_resultados.py
Precisa de numpy, pandas e scipy (o venv do DORL tem). Sai com código 1 se algo diferir ou faltar.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(RAIZ, "results", "logs")
ALE_N4 = "results/e1/aleatoria.json"     # política aleatória na regra N4 (S3)
ALE_E6 = "results/e6/aleatoria_E6.json"  # política aleatória na saída calibrada (E6)
ALE_N8C = "results/e3/aleatoria_N8c.json"  # política aleatória com N = 8 e a janela corrigida
# (prefixo dos arquivos, referência aleatória usada na geração; verificado contra os *_por_semente.csv)
CONJUNTOS = [("results/e1/e1", ALE_N4), ("results/a5/e1", ALE_N4), ("results/e3/e1", ALE_N4),
             ("results/e5/e1", ALE_N4), ("results/e6/e1", ALE_E6), ("results/sb/n4", ALE_N4),
             ("results/sb/e6", ALE_E6), ("results/tmax/n4", None), ("results/tmax/e6", None),
             ("results/s10/n4", ALE_N4), ("results/s10/e6", ALE_E6), ("results/s10/n8c", ALE_N8C)]
# Reexecuções conferidas contra results/e1: rótulo do método em results/e1 → radical da mensagem original
REEXECUCOES = {"MBPO": "MBPO", "MOPO": "MOPO", "DORL": "DORL", "DORL-const": "DORLconst"}
NOME_LOG = re.compile(r"^\[(?P<msg>.+?)\]_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}\.log$")


def glob_do_metodo(msgs):
    """Glob que pega os logs das mensagens de um método, só nas sementes usadas no resumo commitado
    (ex.: MBPO_s0..s4 → [[]MBPO_s[01234][]]_*.log). Logs de outras sementes exportados depois (ex.: s5–9 de um
    resumo com 5 sementes) não entram; os logs descartados (incompletos ou repetidos) das mesmas mensagens entram, como
    na geração original, porque aparecem na lista de logs da saída de texto."""
    radicais = {re.sub(r"_s\d+$", "", m) for m in msgs}
    if len(radicais) != 1:
        raise ValueError(f"mensagens com radicais diferentes: {sorted(radicais)}")
    r = glob.escape(radicais.pop())
    sementes = sorted({mm.group(1) for m in msgs if (mm := re.search(r"_s(\d+)$", m))})
    if any(len(x) > 1 for x in sementes):
        raise ValueError(f"semente com mais de um dígito em {r}: a classe de caracteres do glob não a representa")
    if sementes and len(sementes) == len(set(msgs)):
        return os.path.join(LOGS, f"[[]{r}_s[{''.join(sementes)}][]]_*.log")
    if not sementes:
        return os.path.join(LOGS, f"[[]{r}[]]_*.log")
    return os.path.join(LOGS, f"[[]{r}[]_]*.log")  # ex.: [UCB] (semente 0) e [UCB_s1..4]


def csv_igual(a, b):
    """Mesmas colunas, mesmas linhas, textos iguais e números iguais a 1e-9 (relativo)."""
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return False
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(x[c].to_numpy(float), y[c].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            return False
    return True


def confere_e1(prefixo, aleatoria, tmp):
    base = os.path.join(RAIZ, prefixo)
    ps = pd.read_csv(base + "_por_semente.csv")
    ps = ps[ps["metodo"] != "aleatória"]
    metodos = list(dict.fromkeys(ps["metodo"]))
    logs = ps.drop_duplicates("log")
    faltam = [l for l in logs["log"] if not os.path.exists(os.path.join(LOGS, l))]
    if faltam:
        return f"INCOMPLETO: faltam {len(faltam)} de {len(logs)} logs em results/logs (rodar exporta_resultados.sh na 4090)"
    specs = []
    for m in metodos:
        msgs = [NOME_LOG.match(l).group("msg") for l in logs.loc[logs["metodo"] == m, "log"]]
        specs.append(f"{m}={glob_do_metodo(msgs)}")
    out = os.path.join(tmp, os.path.basename(prefixo))
    cmd = [sys.executable, os.path.join(RAIZ, "e1_protocolos.py"), *specs, "--out", out]
    if aleatoria:
        cmd += ["--aleatoria", os.path.join(RAIZ, aleatoria)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    if r.returncode != 0:
        return f"ERRO ao rodar e1_protocolos.py: {r.stderr.strip().splitlines()[-1]}"
    dif = [s for s in ("_por_semente.csv", "_resumo.csv", "_posicoes.csv")
           if os.path.exists(base + s) and not csv_igual(base + s, out + s)]
    saida = base + "_saida.txt"
    sem_caminho = lambda s: [l for l in s.strip().splitlines() if not l.startswith("Arquivos: ")]  # prefixo do --out
    if os.path.exists(saida) and sem_caminho(open(saida, encoding="utf-8").read()) != sem_caminho(r.stdout):
        dif.append("_saida.txt")
    n = len(logs)
    return f"ok ({len(metodos)} métodos, {n} logs)" if not dif else f"DIFERE em {', '.join(dif)}"


def confere_table2(tmp):
    """As opções só escolhem o que é impresso (protocolos extras; comparação com o artigo), não os valores: são lidas
    do próprio arquivo. A primeira linha (caminho do log na 4090) não entra na comparação."""
    falhas, n = [], 0
    for f in sorted(glob.glob(os.path.join(RAIZ, "results", "table2", "*.txt"))):
        texto = open(f, encoding="utf-8").read()
        linhas = texto.strip().splitlines()
        log = os.path.join(LOGS, os.path.basename(linhas[0].split("Log: ", 1)[1]))
        cmd = [sys.executable, os.path.join(RAIZ, "table2_from_log.py"), log]
        if "| artigo" in texto:
            cmd += ["--dataset", "KuaiRec", "--method", re.sub(r"_s\d+$", "", NOME_LOG.match(os.path.basename(log)).group("msg"))]
        if "== Free (FB)" in texto:
            cmd.append("--all-protocols")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        n += 1
        if r.returncode != 0 or r.stdout.strip().splitlines()[1:] != linhas[1:]:
            falhas.append(os.path.basename(f))
    return f"ok ({n} resumos)" if not falhas else f"DIFERE em {', '.join(falhas)}"


def confere_ope(tmp):
    d = os.path.join(tmp, "ope")
    shutil.copytree(os.path.join(RAIZ, "results", "ope"), d)
    os.remove(os.path.join(d, "ope_comparacao.txt"))
    r = subprocess.run([sys.executable, os.path.join(RAIZ, "ope_compara.py"), "--ope", d, "--e1",
                        os.path.join(RAIZ, "results", "e1", "e1_por_semente.csv")], capture_output=True, text=True)
    if r.returncode != 0:
        return f"ERRO ao rodar ope_compara.py: {r.stderr.strip().splitlines()[-1]}"
    novo = open(os.path.join(d, "ope_comparacao.txt"), encoding="utf-8").read().strip()
    velho = open(os.path.join(RAIZ, "results", "ope", "ope_comparacao.txt"), encoding="utf-8").read().strip()
    return "ok" if novo == velho else "DIFERE"


def confere_reexecucoes(tmp):
    orig = pd.read_csv(os.path.join(RAIZ, "results", "e1", "e1_por_semente.csv"))
    orig["semente"] = orig["log"].str.extract(r"_s(\d+)\]")[0]
    chave = ["metodo", "semente", "protocolo", "metrica"]
    res = []
    for nome, molde in (("trajetórias", "{}_traj"), ("checkpoint", "CK_{}")):
        specs = [f"{rot}={os.path.join(LOGS, '[[]' + glob.escape(molde.format(r)) + '_s[01234][]]_*.log')}"
                 for rot, r in REEXECUCOES.items()]
        out = os.path.join(tmp, "reexec_" + molde.format("x"))
        r = subprocess.run([sys.executable, os.path.join(RAIZ, "e1_protocolos.py"), *specs, "--out", out],
                           capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0:
            res.append(f"{nome}: ERRO ({r.stderr.strip().splitlines()[-1]})")
            continue
        novo = pd.read_csv(out + "_por_semente.csv")
        novo["semente"] = novo["log"].str.extract(r"_s(\d+)\]")[0]
        m = novo.merge(orig, on=chave, suffixes=("", "_orig"))
        esperado = len(REEXECUCOES) * 5 * 3 * 4  # métodos × sementes × protocolos × métricas
        ok = len(novo) == len(m) == esperado and np.allclose(m["valor"], m["valor_orig"], rtol=1e-9, atol=1e-12)
        res.append(f"{nome} {'ok' if ok else 'DIFERE'} ({novo['log'].nunique()} logs)")
    return "; ".join(res)


def main():
    falhou = False
    with tempfile.TemporaryDirectory() as tmp:
        print("1. Resumos entre sementes (e1_protocolos.py)")
        for prefixo, aleatoria in CONJUNTOS:
            res = confere_e1(prefixo, aleatoria, tmp)
            falhou |= not res.startswith("ok")
            print(f"   {prefixo + '_*':22s} {res}")
        res = confere_table2(tmp)
        falhou |= not res.startswith("ok")
        print(f"2. Resumos por log (results/table2): {res}")
        res = confere_ope(tmp)
        falhou |= not res.startswith("ok")
        print(f"3. OPE × simulador (results/ope/ope_comparacao.txt): {res}")
        res = confere_reexecucoes(tmp)
        falhou |= "ok" not in res or "DIFERE" in res or "ERRO" in res
        print(f"4. Reexecuções = execuções originais (results/e1): {res}")
    print("\nTUDO CONFERE" if not falhou else "\nHÁ DIFERENÇAS OU LOGS FALTANDO (ver acima)")
    sys.exit(1 if falhou else 0)


if __name__ == "__main__":
    main()
