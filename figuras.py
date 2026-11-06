"""Gera as figuras de docs/figuras/ a partir de results/ (sem treino, sem dados brutos).

  grafico_E1_rtra.png   E1: posição de cada método em R_tra, NX_0 × NX_10 (results/e1/e1_resumo.csv)
  grafico_S2_top10.png  S2: recompensa de avaliação do top-10 por critério (results/analises/rhat_aval.json)

Uso (na raiz do repositório): python figuras.py [--out docs/figuras]
Só precisa de pandas e matplotlib.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--resumo", default="results/e1/e1_resumo.csv")
p.add_argument("--rhat", default="results/analises/rhat_aval.json")
p.add_argument("--out", default="docs/figuras")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

DESTAQUE = "#2a78d6"     # métodos de RL / o que a política vê
CONTEXTO = "#c3c2b7"     # referências
TEXTO, TEXTO2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
RL = ["MBPO", "MOPO", "DORL", "DORL-const"]
NOMES = {"aleatória": "Aleatória", "eps-greedy": "ε-greedy"}
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 11,
                     "text.color": TEXTO, "savefig.dpi": 200, "figure.facecolor": "white"})


def br(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def titulo(fig, t, sub):
    h = fig.get_figheight()                                   # posições em polegadas a partir do topo
    fig.text(0.03, 1 - 0.18 / h, t, fontsize=14, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / h, sub, fontsize=10.5, color=TEXTO2, va="top")


def espalha(vals, gap):
    """Posições de rótulo: mesma ordem dos valores, com distância mínima `gap` entre vizinhos."""
    ordem = sorted(range(len(vals)), key=lambda i: -vals[i])
    pos = [0.0] * len(vals)
    ult = None
    for i in ordem:
        pos[i] = vals[i] if ult is None else min(vals[i], ult - gap)
        ult = pos[i]
    deslocamento = max(0.0, (max(vals) - max(pos)) / 2 + (min(vals) - min(pos)) / 2)
    return [y + deslocamento for y in pos]


# 1. E1: slope chart de R_tra, NX_0 -> NX_10
r = pd.read_csv(a.resumo)
r = r[r.metrica == "R_tra"].set_index(["protocolo", "metodo"])
metodos = list(r.loc["NX_0"].index)
v0 = {m: r.loc[("NX_0", m), "media"] for m in metodos}
v10 = {m: r.loc[("NX_10", m), "media"] for m in metodos}
n = {m: int(r.loc[("NX_0", m), "n"]) for m in metodos}
pos0 = {m: i + 1 for i, m in enumerate(sorted(metodos, key=lambda m: -v0[m]))}
pos10 = {m: i + 1 for i, m in enumerate(sorted(metodos, key=lambda m: -v10[m]))}
ref = v10["aleatória"] if "aleatória" in v10 else None
abaixo = [m for m in RL if m in v10 and ref is not None and v10[m] < ref]

fig, ax = plt.subplots(figsize=(9.6, 5.4))
fig.subplots_adjust(left=0.30, right=0.72, top=0.76, bottom=0.04)
xs = (0, 1)
for x in xs:
    ax.axvline(x, color=GRADE, lw=1, zorder=0)
for m in sorted(metodos, key=lambda m: m in RL):          # destaque desenhado por cima
    cor = DESTAQUE if m in RL else CONTEXTO
    ax.plot(xs, (v0[m], v10[m]), color=cor, lw=2, zorder=2 + (m in RL), solid_capstyle="round")
    ax.scatter(xs, (v0[m], v10[m]), s=40, color=cor, edgecolor="white", linewidth=1.5, zorder=3 + (m in RL))
span = max(max(v0.values()), max(v10.values())) - min(min(v0.values()), min(v10.values()))
for x, v, pos, ha, dx in ((0, v0, pos0, "right", -0.04), (1, v10, pos10, "left", 0.04)):
    ys = espalha([v[m] for m in metodos], gap=span * 0.055)
    for m, y in zip(metodos, ys):
        cor = TEXTO if m in RL else TEXTO2
        ax.text(x + dx, y, f"{pos[m]}º {NOMES.get(m, m)} {br(v[m])}", ha=ha, va="center", color=cor, fontsize=11)
        if abs(y - v[m]) > span * 0.01:
            ax.plot((x, x + dx * 0.8), (v[m], y), color=GRADE, lw=0.8, zorder=1)
ax.text(0, 1.04, "NX_0 (comprimento livre, Tabela 2)", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
ax.text(1, 1.04, "NX_10 (10 rodadas forçadas)", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.axis("off")
n_rl = sorted({n[m] for m in RL if m in n})
n_ref = ", ".join(f"{NOMES.get(m, m)} {n[m]}" for m in metodos if m not in RL)
quantos = {4: "os quatro", 3: "três", 2: "dois", 1: "um"}.get(len(abaixo), str(len(abaixo)))
t = (f"Com o comprimento fixado em 10, {quantos} métodos de RL ficam abaixo do acaso em R_tra" if abaixo
     else "R_tra por protocolo: comprimento livre × 10 rodadas forçadas")
titulo(fig, t, f"R_tra médio por trajetória · azul: métodos de RL ({'/'.join(map(str, n_rl))} sementes) · "
               f"cinza: referências (sementes: {n_ref})")
fig.savefig(os.path.join(a.out, "grafico_E1_rtra.png"))
plt.close(fig)

# 2. S2: barras horizontais, top-10 por critério
s = json.load(open(a.rhat))
o = s["ordenacao_por_usuario"]
k = s["config"]["k"]
chave = f"recompensa_media_top{k}"
barras = [(r"$\hat{r}$ (o que a política vê)", o["r_hat (o que a política vê)"][chave], DESTAQUE),
          (r"$\hat{r}$ na escala original", o["r_hat na escala original (média + dp do vídeo × r_hat)"][chave], CONTEXTO),
          ("Média do vídeo (sem personalização)", o["só a média do vídeo (sem personalização)"][chave], CONTEXTO),
          ("Oráculo (a própria avaliação)", o["oráculo (recompensa de avaliação)"][chave], CONTEXTO)]
acaso = o["aleatório (média de cada usuário)"][chave]
n_users = s["matriz_avaliacao"]["forma"][0]

fig, ax = plt.subplots(figsize=(9.6, 3.4))
fig.subplots_adjust(left=0.36, right=0.95, top=0.76, bottom=0.04)
ys = range(len(barras))[::-1]
for y, (_, v, cor) in zip(ys, barras):
    ax.barh(y, v, height=0.62, color=cor, zorder=2)
    ax.text(v + 0.06, y, br(v), va="center", fontsize=11)
ax.axvline(acaso, color=TEXTO2, lw=1, ls=(0, (3, 3)), zorder=3)
ax.text(acaso + 0.04, len(barras) - 0.35, f"Aleatório: {br(acaso)}", color=TEXTO2, fontsize=10.5, va="bottom")
ax.set_yticks(list(ys), [b[0] for b in barras], fontsize=11)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xticks([])
for lado in ("top", "right", "bottom"):
    ax.spines[lado].set_visible(False)
ax.spines["left"].set_color(GRADE)
ax.set_xlim(0, max(b[1] for b in barras) * 1.1)
ax.set_ylim(-0.5, len(barras) - 0.2)
pior = barras[0][1] < acaso
titulo(fig, (rf"O $\hat{{r}}$ escolhe itens {'piores' if pior else 'melhores'} que o acaso: "
             f"{br(barras[0][1])} contra {br(acaso)}"),
       f"Recompensa de avaliação média dos {k} primeiros itens de cada usuário "
       f"(watch_ratio, {n_users:,} usuários)".replace(f"{n_users:,}", f"{n_users:,}".replace(",", ".")))
fig.savefig(os.path.join(a.out, "grafico_S2_top10.png"))
plt.close(fig)
print(f"figuras gravadas em {a.out}: grafico_E1_rtra.png, grafico_S2_top10.png")
