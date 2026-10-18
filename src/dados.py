"""Carregamento e preparação dos microdados da PRF.

Variante "agrupados por pessoa, todas as causas e tipos de acidente",
arquivos anuais 2021–2025. Todas as funções recebem/retornam DataFrames
para serem reutilizadas por exploratoria.py e modelos.py.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "base_bruta.parquet"

MACRORREGIAO = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

# Renomeações de causa_acidente dentro de 2021–2025 (grafia e redação).
HARMONIZA_CAUSA = {
    "Transitar no Acostamento": "Transitar no acostamento",
    "Ingestão de álcool e/ou substâncias psicoativas pelo pedestre": "Pedestre: ingestão de álcool ou substâncias",
    "Ingestão de álcool ou de substâncias psicoativas pelo pedestre": "Pedestre: ingestão de álcool ou substâncias",
    "Pedestre - Ingestão de álcool/ substâncias psicoativas": "Pedestre: ingestão de álcool ou substâncias",
}

DTYPES = {
    "id": "int64", "pesid": "float64", "id_veiculo": "float64",
    "br": "float64", "km": "float64", "ano_fabricacao_veiculo": "float64",
    "latitude": str, "longitude": str,
    "idade": "float64", "ilesos": "float32", "feridos_leves": "float32",
    "feridos_graves": "float32", "mortos": "float32", "ordem_tipo_acidente": "float32",
}


def carregar_bruta(anos=range(2021, 2026), usar_cache=True) -> pd.DataFrame:
    """Concatena os arquivos anuais. Mantém latitude/longitude como texto."""
    xz = {ano: DATA_DIR / f"acidentes{ano}_todas_causas_tipos.csv.xz" for ano in anos}
    # Um .xz mais novo que o parquet (a PRF atualiza o ano corrente) refaz o cache.
    if usar_cache and CACHE.exists() and max(
            f.stat().st_mtime for f in xz.values()) <= CACHE.stat().st_mtime:
        df = pd.read_parquet(CACHE)
        return df[df["ano"].isin(list(anos))]
    partes = []
    for ano, f in xz.items():
        d = pd.read_csv(f, sep=";", encoding="ISO-8859-1", decimal=",",
                        dtype=DTYPES, low_memory=False)
        d["ano"] = ano
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")
    if usar_cache:
        df.to_parquet(CACHE)
    return df


def filtrar_analitica(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica os três filtros, nesta ordem. Retorna também o funil."""
    funil = {"base bruta": len(df)}
    df = df[df["causa_principal"] == "Sim"]
    funil["causa_principal == Sim"] = len(df)
    df = df[df["ordem_tipo_acidente"] == 1]
    funil["ordem_tipo_acidente == 1"] = len(df)
    sem_rotulo = df["estado_fisico"].isna() | (df["estado_fisico"] == "Não Informado")
    df = df[~sem_rotulo].copy()
    funil["com rótulo"] = len(df)
    return df, funil


def base_analitica(anos=range(2021, 2026)) -> tuple[pd.DataFrame, dict]:
    """Base filtrada com causas harmonizadas, alvo `y` (1 = óbito) e
    `macrorregiao`. Retorna também o funil de `filtrar_analitica`."""
    df, funil = filtrar_analitica(carregar_bruta(anos))
    df["causa_acidente"] = df["causa_acidente"].astype(str).replace(HARMONIZA_CAUSA)
    df["y"] = (df["estado_fisico"] == "Óbito").astype("int8")
    df["macrorregiao"] = df["uf"].map(MACRORREGIAO)
    assert df["macrorregiao"].notna().all(), "UF sem macrorregião"
    return df, funil
