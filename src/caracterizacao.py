"""Caracterização da base analítica: funil, alvo, regiões, nulos, derivas.

Uso: .venv/bin/python src/caracterizacao.py  -> results/caracterizacao.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dados  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    df, funil = dados.base_analitica()
    out = {"funil": funil}
    out["positivos"] = int(df["y"].sum())
    out["prevalencia"] = float(df["y"].mean())
    out["acidentes"] = int(df["id"].nunique())
    reg = df["macrorregiao"].value_counts()
    out["regioes"] = {k: {"n": int(v), "pct": float(v / len(df))} for k, v in reg.items()}
    out["razao_maior_menor"] = float(reg.max() / reg.min())
    nulos = df[dados.preditores(df, excluir_extra=())].isna().mean().sort_values(ascending=False)
    out["nulos_pct"] = {k: float(v) for k, v in nulos.items() if v > 0}
    out["letalidade_ano"] = {int(k): float(v) for k, v in df.groupby("ano")["y"].mean().items()}
    rep = df[df["ano"].between(2021, 2024)]
    out["recorte_2021_2024"] = {"n": len(rep), "prevalencia": float(rep["y"].mean())}
    for col in ["causa_acidente", "tipo_acidente"]:
        out[f"{col}_distintos"] = {"2021_2024": int(rep[col].nunique()), "2017_2025": int(df[col].nunique())}
    # colisão de id entre anos
    out["ids_em_mais_de_um_ano"] = int((df.groupby("id")["ano"].nunique() > 1).sum())
    # constantes após filtros
    out["constantes"] = {c: int(df[c].nunique()) for c in ["causa_principal", "ordem_tipo_acidente"]}
    # one-hot determinístico do alvo
    out["mortos_vs_obito"] = float(((df["mortos"] > 0) == (df["y"] == 1)).mean())
    out["soma_onehot_igual_1"] = float((df[["ilesos", "feridos_leves", "feridos_graves", "mortos"]].sum(axis=1) == 1).mean())
    # classificacao_acidente vs alvo (nível ocorrência vs pessoa)
    fatal_acc = (df["classificacao_acidente"] == "Com Vítimas Fatais")
    out["classificacao_acidente"] = {
        "P(obito | acidente fatal)": float(df.loc[fatal_acc, "y"].mean()),
        "P(obito | acidente nao fatal)": float(df.loc[~fatal_acc, "y"].mean()),
        "P(acidente fatal)": float(fatal_acc.mean()),
        "cramer_v": float(_cramer_v(pd.crosstab(fatal_acc, df["y"]))),
        "pessoas_por_acidente_fatal_mediana": float(df[fatal_acc].groupby("id").size().median()),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "caracterizacao.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def _cramer_v(ct):
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(ct.values)[0]
    n = ct.values.sum()
    k = min(ct.shape) - 1
    return (chi2 / (n * k)) ** 0.5


if __name__ == "__main__":
    main()
