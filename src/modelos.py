"""XGBoost por macrorregião: desempenho, matriz de transferência 5×5 e SHAP.

Cada região é dividida uma vez em 5 dobras agrupadas por acidente. Na dobra k,
o modelo da região i treina no treino de i (subamostrado para o tamanho do
menor treino, o do Norte, preservando a proporção de classes) e é avaliado no
teste k de cada região j. O limiar de decisão é escolhido num corte interno do
treino e aplicado sem mudança em todas as regiões. Uma segunda rodada, só na
diagonal, inclui os identificadores de lugar, para medir quanto do desempenho
vem de "onde" e não de "o quê".

Uso: .venv/bin/python src/modelos.py [--rapido] [--figuras]
"""
import argparse
import json
import platform
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold)

from dados import base_analitica
import matplotlib
import matplotlib.pyplot as plt

from exploratoria import (ATRIBUTOS, AZUIS, COR, FIG, MARCA, REGIOES, VIRGULA,
                          preparar)

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
SEED = 42
K = 5

CATEG = ["tipo_envolvido", "tipo_acidente_h", "causa_acidente", "tipo_veiculo",
         "tipo_pista", "uso_solo", "fase_dia", "sexo", "condicao_metereologica",
         "sentido_via", "dia_semana"]
NUM = ["idade_n", "idade_veiculo_n", "hora", "ano"]
LUGAR = ["municipio", "br", "delegacia", "km", "latitude", "longitude"]  # só no controle

ROTULO = {**{c: ATRIBUTOS[c] for c in CATEG}, "idade_n": "Idade",
          "idade_veiculo_n": "Idade do veículo", "hora": "Hora", "ano": "Ano",
          "marca": "Marca do veículo", "tracado": "Traçado da via"}
# Rótulo do modelo -> rótulo equivalente no V de Cramér da exploratória.
EQUIV_CRAMER = {"Idade": "Faixa etária", "Hora": "Faixa horária"}

# Busca única, numa amostra nacional; o escolhido vale para todas as regiões.
GRADE = [dict(max_depth=d, min_child_weight=w) for d in (4, 6, 8) for w in (1, 5)]
XGB_FIXO = dict(n_estimators=500, learning_rate=0.1, tree_method="hist",
                enable_categorical=True, early_stopping_rounds=30,
                eval_metric="aucpr", n_jobs=-1)


# ------------------------------------------------------------------ atributos
def montar(df):
    """Matriz de atributos. Mantém `id` (grupo) e `marca` em texto, que é
    codificada por alvo dentro de cada treino."""
    df = preparar(df)
    X = pd.DataFrame({c: df[c].astype(str).astype("category") for c in CATEG})
    X["idade_n"] = df["idade"].where(df["idade"].between(0, 120))
    fab = df["ano_fabricacao_veiculo"].where(df["ano_fabricacao_veiculo"] > 1900)
    X["idade_veiculo_n"] = (df["ano"] - fab).where(lambda s: s >= -1)
    X["hora"] = pd.to_numeric(df["horario"].astype(str).str[:2], errors="coerce")
    X["ano"] = df["ano"]
    trac = df["tracado_via"].astype(str).str.get_dummies(sep=";").add_prefix("tracado_")
    X = pd.concat([X, trac.astype("int8")], axis=1)
    X["marca"] = df["marca"].astype(str)
    for c in ["municipio", "delegacia"]:
        X[c] = df[c].astype(str).astype("category")
    X["br"] = df["br"].astype("Int64").astype(str).astype("category")
    X["km"] = df["km"]
    for c in ["latitude", "longitude"]:
        X[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    X["id"] = df["id"].to_numpy()
    return X.reset_index(drop=True), df["y"].reset_index(drop=True), \
        df["macrorregiao"].reset_index(drop=True)


class GroupTargetEncoder(BaseEstimator, TransformerMixin):
    """Codificação por alvo com suavização (m-estimate) e ajuste fora-da-dobra
    por grupo. No treino, cada linha é codificada com dobras que excluem todo
    o seu acidente (GroupKFold por `id`); sem isso, pessoas do mesmo acidente
    — mesmo br_km, mesmo rótulo — vazam o rótulo umas para as outras. No
    teste, usa-se o mapa ajustado sobre todo o treino. `id` é consumido e
    não sai como atributo."""

    def __init__(self, cols, grupo="id", m=10, n_splits=5):
        self.cols, self.grupo, self.m, self.n_splits = cols, grupo, m, n_splits

    def _mapa(self, X, y):
        prior = y.mean()
        return {c: (y.groupby(X[c]).agg(["sum", "count"]).pipe(lambda t: (t["sum"] + self.m * prior) / (t["count"] + self.m)))
                for c in self.cols}, prior

    def fit(self, X, y):
        self.mapa_, self.prior_ = self._mapa(X, pd.Series(np.asarray(y), index=X.index))
        return self

    def _aplicar(self, X, mapa, prior):
        out = X.drop(columns=[self.grupo]).copy()
        for c in self.cols:
            out[c] = X[c].map(mapa[c]).astype(float).fillna(prior)
        return out

    def transform(self, X):
        return self._aplicar(X, self.mapa_, self.prior_)

    def fit_transform(self, X, y):
        self.fit(X, y)
        y = pd.Series(np.asarray(y), index=X.index)
        out = X.drop(columns=[self.grupo]).copy()
        for c in self.cols:
            out[c] = np.nan
        for tr, te in GroupKFold(self.n_splits).split(X, y, X[self.grupo]):
            mapa, prior = self._mapa(X.iloc[tr], y.iloc[tr])
            enc = self._aplicar(X.iloc[te], mapa, prior)
            out.iloc[te, [out.columns.get_loc(c) for c in self.cols]] = enc[self.cols].values
        return out


# ------------------------------------------------------------------- modelo
MODELOS = {  # os demais classificadores entram aqui no relatório final
    "XGBoost": lambda hp, seed: xgb.XGBClassifier(**XGB_FIXO, **hp, random_state=seed),
}


def melhor_limiar(y, p):
    prec, rev, lim = precision_recall_curve(y, p)
    f1 = 2 * prec * rev / np.maximum(prec + rev, 1e-12)
    return float(lim[np.argmax(f1[:-1])])


def ajustar(nome, X, y, cols, hp, seed):
    """Codificação de `marca` e ajuste num corte interno (80/20 por acidente);
    os 20% servem à parada antecipada e à escolha do limiar."""
    fit, val = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                    .split(X, y, X["id"]))
    enc = GroupTargetEncoder(["marca"])
    A_fit = enc.fit_transform(X.iloc[fit][cols + ["id"]], y.iloc[fit])
    A_val = enc.transform(X.iloc[val][cols + ["id"]])
    clf = MODELOS[nome](hp, seed)
    clf.fit(A_fit, y.iloc[fit], eval_set=[(A_val, y.iloc[val])], verbose=False)
    limiar = melhor_limiar(y.iloc[val], clf.predict_proba(A_val)[:, 1])
    return clf, enc, limiar


def avaliar(clf, enc, limiar, X, y, cols):
    p = clf.predict_proba(enc.transform(X[cols + ["id"]]))[:, 1]
    pred = p >= limiar
    tp = int((pred & (y == 1)).sum())
    prec = tp / max(pred.sum(), 1)
    rev = tp / max((y == 1).sum(), 1)
    return {"f1": 2 * prec * rev / max(prec + rev, 1e-12), "precisao": prec,
            "revocacao": rev, "auc_pr": average_precision_score(y, p),
            "auc_roc": roc_auc_score(y, p), "prevalencia": float(y.mean()),
            "taxa_predita_positiva": float(pred.mean()), "n": len(y)}


def shap_medio(clf, enc, X, cols):
    """|SHAP| médio por atributo (TreeSHAP exato do próprio xgboost).
    Os indicadores de traçado são somados por linha num único atributo."""
    A = enc.transform(X[cols + ["id"]])
    contrib = clf.get_booster().predict(xgb.DMatrix(A, enable_categorical=True),
                                        pred_contribs=True)[:, :-1]
    C = pd.DataFrame(contrib, columns=A.columns)
    trac = [c for c in C if c.startswith("tracado_")]
    C["tracado"] = C[trac].sum(axis=1)
    C = C.drop(columns=trac)
    return C.abs().mean().rename(index=ROTULO)


def subamostrar(y, n, seed):
    """Índices posicionais de `y` com tamanho n e a proporção de classes de y."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    n_pos = int(round(n * len(pos) / len(y)))
    return np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                   rng.choice(neg, n - n_pos, replace=False)]))


# ------------------------------------------------------------------ execução
def buscar_hp(nome, X, y, cols, rapido):
    """Busca única em amostra nacional, CV agrupada de 3 dobras, AUC-PR."""
    idx = subamostrar(y, 20_000 if rapido else 150_000, SEED)
    Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    notas = []
    for hp in (GRADE[:1] if rapido else GRADE):
        s = []
        for tr, te in GroupKFold(3).split(Xs, ys, Xs["id"]):
            clf, enc, _ = ajustar(nome, Xs.iloc[tr], ys.iloc[tr], cols, hp, SEED)
            p = clf.predict_proba(enc.transform(Xs.iloc[te][cols + ["id"]]))[:, 1]
            s.append(average_precision_score(ys.iloc[te], p))
        notas.append({**hp, "auc_pr": float(np.mean(s)), "dp": float(np.std(s))})
        print("  busca", notas[-1], flush=True)
    melhor = max(notas, key=lambda d: d["auc_pr"])
    return {k: melhor[k] for k in GRADE[0]}, notas


def rodar(rapido=False, nome="XGBoost"):
    t0 = time.time()
    df, funil = base_analitica()
    X, y, reg = montar(df)
    del df
    lugar_fora = [c for c in X if c not in LUGAR + ["id"]]
    cols = {"sem_lugar": lugar_fora, "com_lugar": lugar_fora + LUGAR}

    hp, notas = buscar_hp(nome, X, y, cols["sem_lugar"], rapido)
    print("hiperparâmetros:", hp, flush=True)

    por = {r: np.flatnonzero(reg.to_numpy() == r) for r in REGIOES}
    dobras = {r: list(StratifiedGroupKFold(K, shuffle=True, random_state=SEED)
                      .split(por[r], y.iloc[por[r]], X["id"].iloc[por[r]]))
              for r in REGIOES}
    n_sub = min(len(tr) for r in REGIOES for tr, _ in dobras[r])
    if rapido:
        n_sub = 5_000
    linhas, shap = [], []
    for k in range(1 if rapido else K):
        teste = {r: por[r][dobras[r][k][1]] for r in REGIOES}
        for i in REGIOES:
            tr = por[i][dobras[i][k][0]]
            assert not set(X["id"].iloc[tr]) & set(X["id"].iloc[teste[i]]), "id em treino e teste"
            tr = tr[subamostrar(y.iloc[tr], n_sub, SEED + k)]
            Xi, yi = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            for var, c in cols.items():
                clf, enc, lim = ajustar(nome, Xi, yi, c, hp, SEED + k)
                destinos = REGIOES if var == "sem_lugar" else [i]
                for j in destinos:
                    m = avaliar(clf, enc, lim, X.iloc[teste[j]], y.iloc[teste[j]], c)
                    linhas.append({"modelo": nome, "variante": var, "origem": i,
                                   "destino": j, "dobra": k, "limiar": lim,
                                   "arvores": clf.best_iteration + 1, **m})
                if var == "sem_lugar":
                    s = shap_medio(clf, enc, X.iloc[teste[i]], c)
                    shap += [{"modelo": nome, "regiao": i, "dobra": k, "atributo": a,
                              "shap_abs_medio": v} for a, v in s.items()]
            d = next(l for l in reversed(linhas)
                     if l["variante"] == "sem_lugar" and l["destino"] == i)
            print(f"dobra {k} {i:12s} F1 diag {d['f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    sufixo = "_rapido" if rapido else ""
    met = pd.DataFrame(linhas)
    met.to_csv(RES / f"metricas{sufixo}.csv", index=False)
    pd.DataFrame(shap).to_csv(RES / f"shap{sufixo}.csv", index=False)
    cfg = {"semente": SEED, "dobras": K, "n_subamostra_treino": int(n_sub),
           "corte_interno": "GroupShuffleSplit 80/20 por id (parada antecipada e limiar)",
           "codificacao_marca": "GroupTargetEncoder m=10, 5 dobras por id",
           "hiperparametros": {**{k: v for k, v in XGB_FIXO.items()}, **hp},
           "busca": notas, "atributos_sem_lugar": cols["sem_lugar"], "lugar": LUGAR,
           "versoes": {"python": platform.python_version(), "xgboost": xgb.__version__,
                       "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
           "funil": funil, "duracao_s": round(time.time() - t0)}
    (RES / f"modelos{sufixo}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    resumir(sufixo)


# ------------------------------------------------------------------- resumo
def resumir(sufixo=""):
    met = pd.read_csv(RES / f"metricas{sufixo}.csv")
    shap = pd.read_csv(RES / f"shap{sufixo}.csv")
    out = {}
    sem = met[met.variante == "sem_lugar"]
    g = sem.groupby(["origem", "destino"])
    out["matriz_f1"] = {o: {d: [round(g.f1.mean()[o, d], 3), round(g.f1.std()[o, d], 3)]
                            for d in REGIOES} for o in REGIOES}
    for c in ["auc_pr", "auc_roc"]:
        out[f"matriz_{c}"] = {o: {d: [round(g[c].mean()[o, d], 3), round(g[c].std()[o, d], 3)]
                                  for d in REGIOES} for o in REGIOES}
    # Degradação relativa por dobra, depois média. "origem": (M_ii - M_ij)/M_ii,
    # como no plano; confunde-se com a prevalência do destino, que muda o F1.
    # "destino": (M_jj - M_ij)/M_jj, perda do modelo importado frente ao local.
    diag = sem[sem.origem == sem.destino].set_index(["origem", "dobra"])
    for nome, ref, c in [("origem", "origem", "f1"), ("destino", "destino", "f1"),
                         ("destino_auc_pr", "destino", "auc_pr")]:
        base = diag[c].loc[list(zip(sem[ref], sem.dobra))].to_numpy()
        deg = (1 - sem[c] / base).groupby([sem.origem, sem.destino]).agg(["mean", "std"])
        out[f"degradacao_{nome}"] = {o: {d: [round(deg.loc[(o, d), "mean"], 3),
                                             round(deg.loc[(o, d), "std"], 3)]
                                         for d in REGIOES} for o in REGIOES}
    diag_cols = ["f1", "precisao", "revocacao", "auc_pr", "auc_roc", "taxa_predita_positiva", "prevalencia"]
    for var in ["sem_lugar", "com_lugar"]:
        d = met[(met.variante == var) & (met.origem == met.destino)].groupby("origem")[diag_cols]
        out[f"diagonal_{var}"] = {r: {c: [round(d.mean().loc[r, c], 3), round(d.std().loc[r, c], 3)]
                                      for c in diag_cols} for r in REGIOES if r in d.mean().index}

    # Ordenamentos SHAP: por região (média das dobras) e concordância par a par por dobra.
    tab = shap.pivot_table(index=["regiao", "dobra"], columns="atributo", values="shap_abs_medio")
    media = tab.groupby("regiao").mean().T[REGIOES]
    out["shap_medio"] = media.round(4).to_dict()
    out["shap_ordenamento"] = {r: list(media[r].sort_values(ascending=False).index) for r in REGIOES}
    pares = {}
    for a, b in combinations(REGIOES, 2):
        taus, t10, t5 = [], [], []
        for k in tab.index.get_level_values("dobra").unique():
            va, vb = tab.loc[(a, k)], tab.loc[(b, k)]
            taus.append(kendalltau(va, vb)[0])
            t10.append(len(set(va.nlargest(10).index) & set(vb.nlargest(10).index)))
            t5.append(len(set(va.nlargest(5).index) & set(vb.nlargest(5).index)))
        pares[f"{a} x {b}"] = {"tau": [round(np.mean(taus), 3), round(np.std(taus, ddof=1) if len(taus) > 1 else 0, 3)],
                               "top10": float(np.mean(t10)), "top5": float(np.mean(t5))}
    out["shap_pares"] = pares

    # SHAP (condicional) vs. V de Cramér (marginal), nos atributos comuns.
    V = pd.DataFrame(json.loads((RES / "exploratoria.json").read_text())["cramer_v"]).T
    m = media.rename(index=EQUIV_CRAMER)
    comum = [a for a in m.index if a in V.index]
    out["shap_vs_cramer"] = {r: {"tau": round(kendalltau(m.loc[comum, r], V.loc[comum, r])[0], 3),
                                 "n_atributos": len(comum)} for r in REGIOES}
    (RES / f"modelos_resumo{sufixo}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ["matriz_f1", "shap_pares", "shap_vs_cramer"]},
                     ensure_ascii=False, indent=1))
    print("ordenamentos:", json.dumps(out["shap_ordenamento"], ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ figuras
def _heatmap(ax, M, cmap, norm, fmt, escuro):
    v = M.to_numpy()
    ax.imshow(v, cmap=cmap, norm=norm, aspect="auto")
    for a in range(5):
        for b in range(5):
            ax.text(b, a, fmt(v[a, b]), ha="center", va="center",
                    color="white" if escuro(norm(v[a, b])) else "#0b0b0b",
                    fontweight="bold" if a == b else "normal")
    curto = ["N", "NE", "CO", "SE", "S"]
    ax.set_xticks(range(5), curto)
    ax.set_yticks(range(5), REGIOES)
    ax.set_xlabel("Região de avaliação")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def figuras():
    r = json.loads((RES / "modelos_resumo.json").read_text())
    f1 = pd.DataFrame({o: {d: r["matriz_f1"][o][d][0] for d in REGIOES} for o in REGIOES}).T.loc[REGIOES, REGIOES]
    deg = pd.DataFrame({o: {d: r["degradacao_destino"][o][d][0] for d in REGIOES}
                        for o in REGIOES}).T.loc[REGIOES, REGIOES] * 100
    num = lambda f: (lambda x: f.format(x).replace(".", ",").replace("-", "−"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.5))
    azul = matplotlib.colors.LinearSegmentedColormap.from_list("az", AZUIS)
    _heatmap(a1, f1, azul, matplotlib.colors.Normalize(0.2, f1.to_numpy().max()), num("{:.2f}"),
             lambda t: t > 0.55)
    a1.set_ylabel("Região de treino")
    a1.set_title("F1 da classe óbito", loc="left")
    lim = np.abs(deg.to_numpy()).max()
    div = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b02a2a"])
    _heatmap(a2, deg, div, matplotlib.colors.Normalize(-lim, lim), num("{:.0f}"),
             lambda t: abs(t - 0.5) > 0.35)
    a2.set_yticklabels([])
    a2.set_title("Perda frente ao modelo local (%)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "matriz_transferencia.pdf")
    plt.close(fig)

    S = pd.DataFrame(r["shap_medio"])[REGIOES]
    S = S.loc[S.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    y = np.arange(len(S))[::-1]
    ax.hlines(y, S.min(axis=1), S.max(axis=1), color="#c3c2b7", lw=1.2, zorder=1)
    for reg in REGIOES:
        ax.scatter(S[reg], y, s=20, color=COR[reg], marker=MARCA[reg], label=reg,
                   zorder=2, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, S.index)
    ax.set_xlabel("|SHAP| médio (log-chance)")
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.grid(axis="x", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False,
              handletextpad=0.2, columnspacing=1.2, markerscale=1.4)
    fig.tight_layout()
    fig.savefig(FIG / "shap_regioes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    a = ap.parse_args()
    if a.figuras:
        figuras()
    else:
        rodar(a.rapido)
