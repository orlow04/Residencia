"""Gera as figuras de docs/figuras/ a partir de results/ (sem treino, sem dados brutos).

  grafico_E1_rtra.png   E1: posição de cada método em R_tra, NX_0 × NX_10 (results/e1/e1_resumo.csv)
  grafico_S2_top10.png  S2: recompensa de avaliação do top-10 por critério (results/analises/rhat_aval.json)
  grafico_N_length.png  E3/A5: Length NX_0 de MBPO e DORL por N da regra de saída (results/{e1,e3,a5}/e1_resumo.csv)
  grafico_FB_ciclo.png  FB: tamanho do bloco nos ciclos puros contra o ciclo mínimo N + 1 (results/fb/fb_resumo.csv, fb_saida.txt)
  grafico_concentracao.png  MCD (categorias) × cobertura de itens (results/alem/alem_resumo.csv)
  grafico_bolha_V.png   bolha: categorias distintas em 10 itens contra o usuário e o acaso (results/bolha/bolha_resumo.csv)
  grafico_sessoes.png   regra de saída: P(fim | repete) no simulador e nas sessões reais (results/sessoes/sessoes_resumo.csv)
Figura cuja entrada não existe é pulada (com aviso).

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
p.add_argument("--e3", default="results/e3/e1_resumo.csv")
p.add_argument("--a5", default="results/a5/e1_resumo.csv")
p.add_argument("--fb", default="results/fb/fb_resumo.csv")
p.add_argument("--fb_saida", default="results/fb/fb_saida.txt")
p.add_argument("--alem", default="results/alem/alem_resumo.csv")
p.add_argument("--bolha", default="results/bolha/bolha_resumo.csv")
p.add_argument("--sessoes", default="results/sessoes/sessoes_resumo.csv")
p.add_argument("--out", default="docs/figuras")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

DESTAQUE = "#2a78d6"     # métodos de RL / o que a política vê; MBPO nas figuras com dois métodos
SEGUNDA = "#eb6834"      # DORL nas figuras com dois métodos
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
gravadas = ["grafico_E1_rtra.png", "grafico_S2_top10.png"]


def falta(*paths):
    ausentes = [f for f in paths if not os.path.isfile(f)]
    for f in ausentes:
        print(f"aviso: {f} não existe; figura pulada")
    return bool(ausentes)


def eixo_limpo(ax, y=True):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO2, length=0)
    if y:
        ax.grid(axis="y", color=GRADE, lw=0.8, zorder=0)


def resumo_multi(path):
    return pd.read_csv(path, header=[0, 1], index_col=0)


# Configurações da regra de saída: (rótulo, sufixo do nome do método, arquivo); N = 4 original é a reprodução
CONFIGS = [("N = 1", "-N1", a.e3), ("N = 4", "", a.resumo), ("N = 4\ncorrigida", "-N4c", a.a5),
           ("N = 8", "-N8", a.e3), ("N = 8\ncorrigida", "-N8c", a.a5)]
N_DE = {"-N1": 1, "": 4, "-N4c": 4, "-N8": 8, "-N8c": 8}
DOIS = (("MBPO", DESTAQUE), ("DORL", SEGUNDA))


def legenda_dois(ax):
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, markersize=8, label=m) for m, c in DOIS],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10.5)


# 3. E3/A5: Length NX_0 por N
if not falta(a.resumo, a.e3, a.a5):
    e = pd.concat([pd.read_csv(f) for f in {a.resumo, a.e3, a.a5}])
    e = e[(e.protocolo == "NX_0") & (e.metrica == "Length")].drop_duplicates("metodo").set_index("metodo")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.14)
    difs = []
    for x, (rot, suf, _) in enumerate(CONFIGS):
        ys = {}
        for dx, (m, cor) in zip((-0.1, 0.1), DOIS):
            linha = e.loc[m + suf]
            ys[m] = linha.media
            ax.errorbar(x + dx, linha.media, yerr=linha.dp, fmt="o", color=cor, markersize=8, elinewidth=2, capsize=0,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        dif = ys["DORL"] - ys["MBPO"]
        difs.append(dif)
        topo = max(e.loc[m + suf, "media"] + e.loc[m + suf, "dp"] for m, _ in DOIS)
        ax.text(x, topo + 0.6, f"{'+' if dif >= 0 else '−'}{br(abs(dif), 1)}", ha="center", fontsize=10.5, color=TEXTO)
    ax.set_xticks(range(len(CONFIGS)), [c[0] for c in CONFIGS], fontsize=10.5)
    ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
    ax.set_ylabel("Length (passos, NX_0)", color=TEXTO2)
    eixo_limpo(ax)
    legenda_dois(ax)
    corr = [d for d, c in zip(difs, CONFIGS) if c[1] in ("-N1", "-N4c", "-N8c")]   # N ≤ 2 não tem recorte
    ax.set_ylim(None, ax.get_ylim()[1] + 1)
    titulo(fig, f"Sem o recorte da janela, o DORL dura mais em todos os N: +{br(min(corr), 1)} a +{br(max(corr), 1)} passos",
           "Length NX_0, média ± dp entre 5 sementes · número acima de cada par: DORL − MBPO · "
           "\"corrigida\": janela com os min(t, N) itens anteriores")
    fig.savefig(os.path.join(a.out, "grafico_N_length.png"))
    plt.close(fig)
    gravadas.append("grafico_N_length.png")

def blocos_fb(path):
    """Tabela "fração com cada tamanho de bloco" do fb_saida.txt (o fb_resumo.csv não a traz)."""
    linhas = [l.split() for l in open(path).read().split("Entre as trajetórias em ciclo puro")[1].splitlines()[1:] if l.strip()]
    return pd.DataFrame([list(map(float, l[1:])) for l in linhas[1:]], index=[l[0] for l in linhas[1:]],
                        columns=list(map(int, linhas[0])))


# 4. FB: tamanho do bloco nos ciclos puros contra N + 1
if not falta(a.fb, a.fb_saida):
    f, b = resumo_multi(a.fb), blocos_fb(a.fb_saida)
    fig, axs = plt.subplots(2, len(CONFIGS), figsize=(9.6, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.11, wspace=0.12, hspace=0.42)
    domina = []
    for lin, (m, cor) in enumerate(DOIS):
        for col, (rot, suf, _) in enumerate(CONFIGS):
            ax, alvo = axs[lin, col], N_DE[suf] + 1
            fr = b.loc[m + suf]
            ax.bar(fr.index, fr.values, width=0.72, color=[cor if p == alvo else CONTEXTO for p in fr.index], zorder=2)
            topo = fr.reindex([alvo - 1, alvo, alvo + 1]).max()               # acima das barras vizinhas
            ax.text(alvo, topo + 0.03, f"{100 * fr[alvo]:.0f}%", ha="center", fontsize=9.5, color=TEXTO)
            puro, dp = f.loc[m + suf, ("media", "ciclo_puro")], f.loc[m + suf, ("dp", "ciclo_puro")]
            ax.set_title(f"{puro * 100:.0f} ± {dp * 100:.0f}% em ciclo puro", fontsize=9.5, color=TEXTO2, pad=4)
            if lin == 0:
                ax.text(0.5, 1.22, rot.replace("\n", " "), transform=ax.transAxes, ha="center", fontsize=10.5)
                domina.append(fr[alvo] > 0.5)
            ax.set_xticks([1, alvo, 10] if alvo < 9 else [1, 5, alvo])
            ax.set_ylim(0, 1.15)
            ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
            eixo_limpo(ax)
            ax.tick_params(labelsize=9.5)
        axs[lin, 0].set_ylabel(m, fontsize=11, color=TEXTO, fontweight="bold")
    fig.text(0.535, 0.02, "Tamanho do bloco repetido (itens) · barra colorida: N + 1", ha="center", fontsize=10.5, color=TEXTO2)
    onde = "em todas as configurações" if all(domina) else f"em {sum(domina)} das {len(domina)} configurações"
    titulo(fig, f"Quando o MBPO entra em ciclo no FB, o bloco tem N + 1 itens {onde}",
           "Fração dos ciclos puros por tamanho do bloco, média entre 5 sementes · acima de cada painel: "
           "trajetórias FB em ciclo puro (± dp)")
    fig.savefig(os.path.join(a.out, "grafico_FB_ciclo.png"))
    plt.close(fig)
    gravadas.append("grafico_FB_ciclo.png")

# 5. MCD × cobertura de itens
if not falta(a.alem):
    r = resumo_multi(a.alem)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.12)
    # rótulo de cada ponto de MBPO/MOPO: (alinhamento, dx, fator em y); a família do DORL fica num rótulo só
    LADO = {"MBPO": ("center", 0, 1.13), "MBPO-N4c": ("right", -0.014, 0.97), "MBPO-N8c": ("right", -0.014, 0.97)}
    dorl = [m for m in r.index if m.startswith("DORL")]
    mcd_d = r.loc[dorl, ("media", "MCD")]
    fora = [m for m in dorl if mcd_d[m] > mcd_d.median() + 0.15]          # longe do grupo: rótulo próprio (DORL-E6)
    LADO.update({m: ("right", -0.014, 0.97) for m in fora})
    for m in r.index:
        x, y = r.loc[m, ("media", "MCD")], r.loc[m, ("media", "cobertura")]
        cor = CONTEXTO if m == "aleatória" else SEGUNDA if m in dorl else DESTAQUE
        ax.scatter(x, y, s=70, color=cor, edgecolor="white", linewidth=2, zorder=3)
        if m not in dorl or m in fora:
            ha, dx, fy = LADO.get(m, ("left", 0.014, 0.97))
            ax.text(x + dx, y * fy, NOMES.get(m, m), fontsize=10, ha=ha, color=TEXTO2 if cor == CONTEXTO else TEXTO)
    grupo = [m for m in dorl if m not in fora]
    if grupo:
        c = r.loc[grupo, ("media", "cobertura")]
        ax.text(r.loc[grupo, ("media", "MCD")].max() + 0.02, c.mean() * 0.97,
                f"DORL, DORL-const e variantes ({len(grupo)}): {br(100 * c.min(), 1)}% a {br(100 * c.max(), 1)}%",
                fontsize=10, color=TEXTO)
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1], ["2%", "5%", "10%", "20%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.017, 1.15)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1], [br(v, 1) for v in (0.2, 0.4, 0.6, 0.8, 1)])
    ax.set_xlabel("MCD: fração dos itens em categorias dominantes (menor = menos concentrado)", color=TEXTO2)
    ax.set_ylabel("Cobertura: itens distintos por época (escala log)", color=TEXTO2)
    eixo_limpo(ax)
    ax.grid(axis="x", color=GRADE, lw=0.8, zorder=0)
    titulo(fig, "Pela MCD o RL concentra menos que o acaso; pela cobertura de itens, muito mais",
           "Trajetórias NX_10, média entre 5 sementes · azul: MBPO, MOPO e variantes · laranja: DORL e variantes · "
           "cinza: aleatória")
    fig.savefig(os.path.join(a.out, "grafico_concentracao.png"))
    plt.close(fig)
    gravadas.append("grafico_concentracao.png")

# 6. Bolha: V por método contra o usuário e o acaso
if not falta(a.bolha):
    r = resumo_multi(a.bolha)
    metodos = [m for m in r.index if m != "aleatória"]
    vu = r[("media", "V_u")].mean()
    va = r.loc["aleatória", ("media", "V")] if "aleatória" in r.index else None
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(metodos) + 1.9))
    fig.subplots_adjust(left=0.17, right=0.95, top=1 - 1.0 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    ys = range(len(metodos))[::-1]
    for y, m in zip(ys, metodos):
        ax.errorbar(r.loc[m, ("media", "V")], y, xerr=r.loc[m, ("dp", "V")], fmt="o", color=DESTAQUE, markersize=8,
                    elinewidth=2, capsize=0, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    for v, rot in ((vu, "usuário"), (va, "aleatória")):
        if v is not None:
            ax.axvline(v, color=TEXTO2, lw=1, ls=(0, (3, 3)), zorder=2)
            ax.text(v, len(metodos) - 0.4, f"{rot}: {br(v)}", ha="center", va="bottom", fontsize=10, color=TEXTO2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.set_yticks(list(ys), metodos, fontsize=10.5)
    ax.set_ylim(-0.6, len(metodos) - 0.1)
    ax.set_xlabel("V: categorias distintas em 10 itens (NX_10)", color=TEXTO2)
    eixo_limpo(ax, y=False)
    ax.grid(axis="x", color=GRADE, lw=0.8, zorder=0)
    titulo(fig, "Nenhum método de RL fecha o usuário: todos variam mais categorias que ele e que o acaso",
           "Média ± dp entre 5 sementes · usuário: janelas de 10 itens do próprio usuário no big_matrix")
    fig.savefig(os.path.join(a.out, "grafico_bolha_V.png"))
    plt.close(fig)
    gravadas.append("grafico_bolha_V.png")

# 7. Regra de saída: simulador × sessões reais
if not falta(a.sessoes):
    r = pd.read_csv(a.sessoes)
    lin = r[(r.folga_s == 2) & (r.N == 1) & (r.G_min == 30)].iloc[0]
    grupos = [("Simulador (regra do DORL)", 1.0, 0.0), ("Sessões reais (big_matrix)", lin.P_fim_repete, lin.P_fim_nao_repete)]
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    fig.subplots_adjust(left=0.27, right=0.93, top=0.70, bottom=0.06)
    for i, (rot, rep, nao) in enumerate(grupos):
        y = len(grupos) - 1 - i
        for dy, v, cor in ((0.17, rep, DESTAQUE), (-0.17, nao, SEGUNDA)):
            ax.barh(y + dy, v, height=0.3, color=cor, zorder=2)
            ax.text(v + 0.012, y + dy, f"{br(100 * v, 1)}%", va="center", fontsize=10.5)
    ax.set_yticks(range(len(grupos))[::-1], [g[0] for g in grupos], fontsize=10.5)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    eixo_limpo(ax, y=False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=DESTAQUE, label="o item repete categoria do anterior"),
                       plt.Rectangle((0, 0), 1, 1, color=SEGUNDA, label="não repete")],
              frameon=False, loc="lower right", fontsize=10)
    titulo(fig, f"Nos dados reais, repetir categoria aumenta a saída ({br(100 * lin.P_fim_repete, 1)}% contra "
                f"{br(100 * lin.P_fim_nao_repete, 1)}%), mas não a torna certa",
           f"P(sessão termina após o item), N = 1 · sessões reais: cadeias contíguas, fim = {lin.G_min:g} min sem "
           f"atividade, {int(lin.linhas_elegiveis):,} itens".replace(f"{int(lin.linhas_elegiveis):,}",
                                                                    f"{int(lin.linhas_elegiveis):,}".replace(",", ".")))
    fig.savefig(os.path.join(a.out, "grafico_sessoes.png"))
    plt.close(fig)
    gravadas.append("grafico_sessoes.png")

print(f"figuras gravadas em {a.out}: {', '.join(gravadas)}")
