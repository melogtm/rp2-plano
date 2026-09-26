"""Análise exploratória da base analítica (2017–2025) por macrorregião.

Gera as figuras do relatório em figuras/ e os números citados no texto em
results/exploratoria.json. Uso: .venv/bin/python src/exploratoria.py
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kendalltau

from dados import base_analitica

RAIZ = Path(__file__).resolve().parent.parent
FIG = RAIZ / "figuras"
RES = RAIZ / "results"
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
# Paleta categórica de referência (ordem fixa, validada para daltonismo).
COR = dict(zip(REGIOES, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]))
# Forma por região: identidade não depende só da cor (sem validador de CVD aqui).
MARCA = dict(zip(REGIOES, ["o", "s", "^", "D", "v"]))
AZUIS = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Renomeações de tipo_acidente em 2020–2021 (fusão/partição de categorias).
HARMONIZA_TIPO = {
    "Colisão lateral mesmo sentido": "Colisão lateral",
    "Colisão lateral sentido oposto": "Colisão lateral",
    "Colisão com objeto estático": "Colisão com objeto",
    "Colisão com objeto em movimento": "Colisão com objeto",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42,
})


VIRGULA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def afastar(valores, gap):
    """Posições de rótulo com distância mínima `gap`, na ordem dos valores."""
    ordem = sorted(valores, key=valores.get)
    pos, ult = {}, -np.inf
    for k in ordem:
        pos[k] = max(valores[k], ult + gap)
        ult = pos[k]
    return pos


def preparar(df):
    df["tipo_acidente_h"] = df["tipo_acidente"].astype(str).replace(HARMONIZA_TIPO)
    idade = df["idade"].where(df["idade"].between(0, 120))
    df["faixa_etaria"] = pd.cut(idade, [-1, 17, 29, 44, 59, 120],
                                labels=["0-17", "18-29", "30-44", "45-59", "60+"])
    fab = df["ano_fabricacao_veiculo"].where(df["ano_fabricacao_veiculo"] > 1900)
    df["idade_veiculo"] = pd.cut(df["ano"] - fab, [-2, 5, 10, 20, 200],
                                 labels=["0-5", "6-10", "11-20", "21+"])
    hora = pd.to_numeric(df["horario"].astype(str).str[:2], errors="coerce")
    df["faixa_horaria"] = pd.cut(hora, [-1, 5, 11, 17, 23],
                                 labels=["0-5h", "6-11h", "12-17h", "18-23h"])
    # tracado_via concatena descritores em ordem arbitrária; forma canônica.
    df["tracado_via_c"] = (df["tracado_via"].astype(str).str.split(";")
                           .map(lambda d: ";".join(sorted(d))))
    return df


def cramer_v(x, y):
    """V de Cramér com correção de viés de Bergsma (2013)."""
    t = pd.crosstab(x, y)
    if t.shape[0] < 2:
        return 0.0
    n = t.to_numpy().sum()
    phi2 = chi2_contingency(t, correction=False)[0] / n
    r, k = t.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(kc - 1, rc - 1)))


ATRIBUTOS = {  # coluna -> rótulo no gráfico
    "tipo_envolvido": "Tipo de envolvido",
    "tipo_acidente_h": "Tipo de acidente",
    "causa_acidente": "Causa do acidente",
    "tipo_veiculo": "Tipo de veículo",
    "tipo_pista": "Tipo de pista",
    "uso_solo": "Uso do solo",
    "fase_dia": "Fase do dia",
    "faixa_horaria": "Faixa horária",
    "tracado_via_c": "Traçado da via",
    "sexo": "Sexo",
    "faixa_etaria": "Faixa etária",
    "idade_veiculo": "Idade do veículo",
    "condicao_metereologica": "Condição meteorológica",
    "sentido_via": "Sentido da via",
    "dia_semana": "Dia da semana",
}


def fig_letalidade_ano(df):
    tab = df.pivot_table(index="ano", columns="macrorregiao", values="y",
                         aggfunc="mean", observed=True) * 100
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    nac = df.groupby("ano").y.mean() * 100
    fim = {r: tab[r].iloc[-1] for r in REGIOES} | {"Brasil": nac.iloc[-1]}
    pos = afastar(fim, 0.2)
    for r in REGIOES:
        ax.plot(tab.index, tab[r], color=COR[r], lw=1.6, marker=MARCA[r], ms=3.5)
    ax.plot(nac.index, nac, color="#8a8984", lw=1.2, ls="--")
    for r, yv in pos.items():
        ax.text(tab.index[-1] + 0.15, yv, r, va="center",
                color="#52514e" if r == "Brasil" else "#0b0b0b")
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.set_xticks(tab.index)
    ax.set_xlim(2016.7, 2026.4)
    ax.set_ylabel("Letalidade por envolvido (%)")
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "letalidade_ano.pdf")
    plt.close(fig)
    return tab.round(2)


LINHAS_HEATMAP = [  # (coluna, categoria, rótulo)
    ("tipo_envolvido", "Pedestre", "Envolvido: pedestre"),
    ("tipo_acidente_h", "Atropelamento de Pedestre", "Atropelamento de pedestre"),
    ("tipo_acidente_h", "Colisão frontal", "Colisão frontal"),
    ("tipo_veiculo", "Bicicleta", "Veículo: bicicleta"),
    ("tipo_veiculo", "Motocicleta", "Veículo: motocicleta"),
    ("fase_dia", "Plena Noite", "Plena noite"),
    ("tipo_pista", "Simples", "Pista simples"),
    ("uso_solo", "Não", "Área rural"),
    ("tipo_acidente_h", "Saída de leito carroçável", "Saída de leito carroçável"),
    ("tipo_veiculo", "Automóvel", "Veículo: automóvel"),
    ("tipo_pista", "Dupla", "Pista dupla"),
    ("tipo_acidente_h", "Colisão traseira", "Colisão traseira"),
]


def fig_letalidade_categorias(df):
    let, part = {}, {}
    for col, cat, rot in LINHAS_HEATMAP:
        m = df[col].astype(str) == cat
        let[rot] = df[m].groupby("macrorregiao", observed=True).y.mean() * 100
        part[rot] = m.groupby(df["macrorregiao"], observed=True).mean() * 100
    let["Todos os envolvidos"] = df.groupby("macrorregiao", observed=True).y.mean() * 100
    let = pd.DataFrame(let).T[REGIOES]
    part = pd.DataFrame(part).T[REGIOES]

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    v = let.to_numpy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("az", AZUIS)
    norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=np.nanmax(v))
    ax.imshow(v, cmap=cmap, norm=norm, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            escuro = norm(v[i, j]) > 0.55
            ax.text(j, i, f"{v[i, j]:.1f}".replace(".", ","), ha="center",
                    va="center", color="white" if escuro else "#0b0b0b")
    ax.set_xticks(range(len(REGIOES)), REGIOES)
    ax.set_yticks(range(len(let)), let.index)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.axhline(len(let) - 1.5, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "letalidade_categorias.pdf")
    plt.close(fig)
    return let.round(2), part.round(2)


def associacao(df):
    V = pd.DataFrame({r: {rot: cramer_v(d[c].astype(str), d["y"])
                          for c, rot in ATRIBUTOS.items()}
                      for r, d in df.groupby("macrorregiao", observed=True)})[REGIOES]
    V = V.loc[V.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = np.arange(len(V))[::-1]
    ax.hlines(y, V.min(axis=1), V.max(axis=1), color="#c3c2b7", lw=1.2, zorder=1)
    for r in REGIOES:
        ax.scatter(V[r], y, s=20, color=COR[r], marker=MARCA[r], label=r, zorder=2,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, V.index)
    ax.set_xlabel("V de Cramér com o óbito do envolvido")
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.grid(axis="x", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.4, 1.0),
              frameon=False, handletextpad=0.2, columnspacing=1.2, markerscale=1.4)
    fig.tight_layout()
    fig.savefig(FIG / "associacao_regioes.pdf")
    plt.close(fig)

    ranks = V.rank(ascending=False)
    pares = {}
    for a, b in combinations(REGIOES, 2):
        top5 = len(set(V[a].nlargest(5).index) & set(V[b].nlargest(5).index))
        pares[f"{a} x {b}"] = {"tau": round(kendalltau(ranks[a], ranks[b])[0], 3),
                               "top5_comum": top5}
    return V.round(4), ranks, pares


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    df, funil = base_analitica()
    df = preparar(df)
    out = {"funil": funil}

    reg = df.groupby("macrorregiao", observed=True).agg(
        registros=("y", "size"), acidentes=("id", "nunique"), obitos=("y", "sum"),
        letalidade=("y", "mean"))
    reg["pct_registros"] = reg.registros / len(df) * 100
    reg["letalidade"] *= 100
    reg["pct_moto"] = df.groupby("macrorregiao", observed=True).tipo_veiculo.apply(
        lambda s: (s == "Motocicleta").mean() * 100)
    reg["pct_pista_simples"] = df.groupby("macrorregiao", observed=True).tipo_pista.apply(
        lambda s: (s == "Simples").mean() * 100)
    reg["pct_rural"] = df.groupby("macrorregiao", observed=True).uso_solo.apply(
        lambda s: (s == "Não").mean() * 100)
    out["regioes"] = reg.loc[REGIOES].round(2).to_dict(orient="index")
    out["brasil"] = {"registros": len(df), "acidentes": int(df.id.nunique()),
                     "obitos": int(df.y.sum()), "letalidade": round(df.y.mean() * 100, 3)}

    out["letalidade_ano"] = fig_letalidade_ano(df).to_dict()
    out["letalidade_ano_brasil"] = (df.groupby("ano").y.mean() * 100).round(2).to_dict()
    let, part = fig_letalidade_categorias(df)
    out["letalidade_categorias"] = let.to_dict(orient="index")
    out["participacao_categorias"] = part.to_dict(orient="index")

    # Associação: 2021–2025, período com taxonomia única de causa_acidente.
    rec = df[df["ano"] >= 2021]
    V, ranks, pares = associacao(rec)
    out["associacao_periodo"] = {"anos": "2021-2025", "registros": len(rec)}
    out["cramer_v"] = V.to_dict(orient="index")
    out["kendall_tau_pares"] = pares

    cat = pd.crosstab(df["causa_acidente"], df["ano"]) > 0
    out["categorias_por_ano"] = {
        "causa_acidente": cat.sum().to_dict(),
        "causa_so_ate_2020": int((cat.loc[:, 2021:].sum(axis=1) == 0).sum()),
        "causa_so_desde_2021": int((cat.loc[:, :2020].sum(axis=1) == 0).sum()),
        "tipo_acidente_bruto": int(df["tipo_acidente"].nunique()),
        "tipo_acidente_harmonizado": int(df["tipo_acidente_h"].nunique()),
        "tracado_via_bruto": int(df["tracado_via"].nunique()),
        "tracado_via_canonico": int(df["tracado_via_c"].nunique()),
        "tracado_via_um_descritor": round(
            (~df["tracado_via"].astype(str).str.contains(";")).mean() * 100, 2),
    }
    (RES / "exploratoria.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ["regioes", "kendall_tau_pares",
                                          "categorias_por_ano", "cramer_v"]},
                     ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
