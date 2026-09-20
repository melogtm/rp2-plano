"""Agrega results/replicacao_*.csv em média ± desvio-padrão por configuração e modelo."""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
METRICAS = ["f1", "precisao", "revocacao", "auc_pr", "auc_roc"]


def agregar() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(RESULTS / f"replicacao_{c}.csv") for c in ("publicado", "corrigido")])
    g = df.groupby(["config", "modelo"])[METRICAS]
    return g.agg(["mean", "std"]).round(4), df


if __name__ == "__main__":
    tab, _ = agregar()
    pd.set_option("display.width", 200)
    print(tab)
    tab.to_csv(RESULTS / "replicacao_resumo.csv")
    for (cfg, m), r in tab.iterrows():
        print(cfg, m, " & ".join(f"{r[(k,'mean')]:.3f} $\\pm$ {r[(k,'std')]:.3f}" for k in METRICAS), r"\\")
