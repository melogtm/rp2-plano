"""Replicação do estudo de referência (Gabardo et al., 2025) em duas configurações.

"publicado": reproduz deliberadamente D1, D2 e D3 conforme train_models.R.
"corrigido": StratifiedGroupKFold por id, codificação por alvo e SMOTE
             dentro do Pipeline, ajustados só na partição de treino.

Recorte: região Sul, 2021–2024, alvo e filtros do estudo de referência
(gravidade = classificacao_acidente == "Com Vítimas Fatais").

Uso: .venv/bin/python src/replicacao.py [publicado|corrigido] -> results/
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).parent))
import dados  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 420          # mesma semente do script original (set.seed(420))
N_REP = 5
SMOTE_K = 3         # SMOTE(..., K = 3, dup_size = 12)
SMOTE_DUP = 12
CATEGORICAS = ["dia_semana", "tipo_acidente", "fase_dia", "sentido_via",
               "condicao_metereologica", "tipo_pista", "uso_solo",
               "tipo_veiculo", "sexo", "br_km"]
NUMERICAS = ["ano_fabricacao_veiculo", "idade"]

# Valores selecionados pela busca em grade do estudo não são reportados;
# usam-se os padrões das bibliotecas (ranger / scikit-learn), documentados
# em results/replicacao_config.json. MLP: max_iter=100 espelha maxit=100 do nnet;
# batch_size=1024 apenas por custo (nnet usa lote completo).
HIPER = {
    "kNN": dict(n_neighbors=5),
    "RF": dict(n_estimators=500, max_features="sqrt", min_samples_leaf=1,
               criterion="gini", n_jobs=-1),
    "MLP": dict(hidden_layer_sizes=(100,), alpha=1e-4, max_iter=100, batch_size=1024),
}


def modelo(nome, seed):
    if nome == "kNN":
        return KNeighborsClassifier(**HIPER["kNN"], n_jobs=-1)
    if nome == "RF":
        return RandomForestClassifier(**HIPER["RF"], random_state=seed)
    return MLPClassifier(**HIPER["MLP"], random_state=seed)


# ---------------------------------------------------------------- dados
TIPO_VEICULO = {
    **dict.fromkeys(["Camioneta", "Caminhonete", "Caminhão-trator", "Caminhão",
                     "Reboque", "Semireboque", "Trator de rodas"], "Caminhão"),
    **dict.fromkeys(["Ônibus", "Micro-ônibus"], "Ônibus"),
    **dict.fromkeys(["Motocicleta", "Motoneta"], "Motocicleta"),
    **dict.fromkeys(["Chassi-plataforma", "Ciclomotor", "Triciclo", "Outros"], "Outros"),
    **dict.fromkeys(["Automóvel", "Utilitário"], "Carro"),
}
TIPO_ACIDENTE = {
    **dict.fromkeys(["Colisão traseira", "Colisão com objeto", "Colisão frontal"], "Colisão"),
    **dict.fromkeys(["Colisão lateral sentido oposto", "Colisão lateral mesmo sentido",
                     "Colisão lateral", "Colisão transversal"], "Colisão lateral"),
    **dict.fromkeys(["Atropelamento de Pedestre", "Atropelamento de Animal"], "Atropelamento"),
    **dict.fromkeys(["Eventos atípicos", "Queda de ocupante de veículo",
                     "Derramamento de carga"], "Eventos atípicos"),
}


def recorte_replicacao() -> pd.DataFrame:
    """Sul 2021–2024, deduplicado (filtros 1 e 2), filtros e alvo do estudo."""
    df = dados.carregar_bruta(anos=range(2021, 2025))
    df = df[(df["causa_principal"] == "Sim") & (df["ordem_tipo_acidente"] == 1)]
    df = df[df["uf"].isin(["PR", "SC", "RS"])]
    cols = ["id", "classificacao_acidente", "br", "km"] + CATEGORICAS[:-1] + NUMERICAS
    df = df[cols].copy()
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].astype(object)
    df = df.dropna()                                        # na.omit
    df = df[(df["ano_fabricacao_veiculo"] != 0) & df["idade"].between(1, 120)]
    df = df[df["sexo"].isin(["Masculino", "Feminino"])]
    df = df[df["condicao_metereologica"] != "Ignorado"]
    df = df[df["sentido_via"] != "Não Informado"]
    df = df[df["tipo_veiculo"] != "Não Informado"]
    df["y"] = (df["classificacao_acidente"] == "Com Vítimas Fatais").astype(int)
    df["br_km"] = df["br"].astype(int).astype(str) + "_" + (np.round(df["km"] / 5) * 5).astype(int).astype(str)
    df["tipo_veiculo"] = df["tipo_veiculo"].replace(TIPO_VEICULO)
    df = df[df["tipo_veiculo"] != "Outros"]
    df["tipo_acidente"] = df["tipo_acidente"].replace(TIPO_ACIDENTE)
    return df.drop(columns=["classificacao_acidente", "br", "km"]).reset_index(drop=True)


# ------------------------------------------------------- config publicado
def kfold_encode_original(df, y, cols, k, rng):
    """Transcrição de kfold_encode(): a cada dobra o mapa é recalculado sobre a
    própria dobra e sobrescreve a coluna inteira; o resultado final é o mapa
    da última dobra, aplicado a todas as linhas (inclusive as dela).
    Categorias ausentes recebem a média global do alvo. Sem suavização."""
    folds = list(StratifiedKFold(k, shuffle=True, random_state=rng).split(df, y))
    out = df.copy()
    for var in cols:
        for _, idx in folds:
            mapa = y.iloc[idx].groupby(df[var].iloc[idx]).mean()
            out[var] = df[var].map(mapa).fillna(y.mean())
    return out


def rodar_publicado(df, seed):
    """D1+D2+D3 como em train_models.R, uma repetição por semente."""
    rng = np.random.RandomState(seed)
    y = df["y"]
    X = kfold_encode_original(df[CATEGORICAS + NUMERICAS], y, CATEGORICAS, 5, seed)   # D2
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)          # scale() na base completa
    # D1: createDataPartition estratificado 80%; df <- df[idx]; df.test <- df[-idx]
    idx = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], int(round(0.8 * (y == c).sum())), replace=False)
                                  for c in (0, 1)]))
    X_tr, y_tr = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    pos_teste = np.setdiff1d(np.arange(len(X_tr)), idx)      # índices negativos sobre o df já reduzido
    X_te, y_te = X_tr.iloc[pos_teste], y_tr.iloc[pos_teste]
    assert len(pos_teste) > 0
    # D3: SMOTE sobre o treino (que contém o teste), K=3, 12 sintéticas por positivo
    razao = min(1.0, (1 + SMOTE_DUP) * y_tr.sum() / (len(y_tr) - y_tr.sum()))
    X_sm, y_sm = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=razao, random_state=seed).fit_resample(X_tr, y_tr)
    return X_sm, y_sm, X_te, y_te, {"n_treino": len(y_tr), "n_treino_smote": len(y_sm),
                                    "n_teste": len(y_te), "teste_contido_no_treino": True}


# ------------------------------------------------------- config corrigido
def pipeline_corrigido(nome, seed, razao):
    return Pipeline([
        ("te", TargetEncoder(target_type="binary", smooth="auto", random_state=seed)),
        ("scale", StandardScaler()),
        ("smote", SMOTE(k_neighbors=SMOTE_K, sampling_strategy=razao, random_state=seed)),
        ("clf", modelo(nome, seed)),
    ])


def metricas(y_true, prob, pred):
    return dict(f1=f1_score(y_true, pred), precisao=precision_score(y_true, pred, zero_division=0),
                revocacao=recall_score(y_true, pred), auc_pr=average_precision_score(y_true, prob),
                auc_roc=roc_auc_score(y_true, prob))


def main(config):
    df = recorte_replicacao()
    razao = min(1.0, (1 + SMOTE_DUP) * df["y"].sum() / (len(df) - df["y"].sum()))
    print(f"recorte: n={len(df)} positivos={df['y'].sum()} ({df['y'].mean():.4%}) acidentes={df['id'].nunique()}")
    linhas = []
    if config == "publicado":
        for rep in range(N_REP):
            seed = SEED + rep
            X_sm, y_sm, X_te, y_te, info = rodar_publicado(df, seed)
            for nome in HIPER:
                t = time.time()
                m = modelo(nome, seed).fit(X_sm, y_sm)
                prob = m.predict_proba(X_te)[:, 1]
                linhas.append(dict(config=config, modelo=nome, dobra=rep, seed=seed, **info,
                                   **metricas(y_te, prob, (prob >= 0.5).astype(int)), tempo_s=time.time() - t))
                print(linhas[-1], flush=True)
    else:
        X, y, g = df[CATEGORICAS + NUMERICAS], df["y"], df["id"]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (tr, te) in enumerate(cv.split(X, y, g)):
            assert not set(g.iloc[tr]) & set(g.iloc[te])
            for nome in HIPER:
                t = time.time()
                p = pipeline_corrigido(nome, SEED + k, razao).fit(X.iloc[tr], y.iloc[tr])
                prob = p.predict_proba(X.iloc[te])[:, 1]
                linhas.append(dict(config=config, modelo=nome, dobra=k, seed=SEED + k, n_treino=len(tr),
                                   n_teste=len(te), teste_contido_no_treino=False,
                                   **metricas(y.iloc[te], prob, (prob >= 0.5).astype(int)), tempo_s=time.time() - t))
                print(linhas[-1], flush=True)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(linhas).to_csv(RESULTS / f"replicacao_{config}.csv", index=False)
    cfg = dict(seed=SEED, n_rep=N_REP, smote=dict(k_neighbors=SMOTE_K, dup_size=SMOTE_DUP, razao_efetiva=razao),
               hiperparametros=HIPER, recorte=dict(n=len(df), positivos=int(df["y"].sum()),
               prevalencia=float(df["y"].mean()), acidentes=int(df["id"].nunique()), regiao="Sul", anos="2021-2024",
               alvo="classificacao_acidente == 'Com Vítimas Fatais'"),
               versoes=dict(python=platform.python_version(), sklearn=sklearn.__version__, imblearn=imblearn.__version__,
                            pandas=pd.__version__, numpy=np.__version__))
    (RESULTS / "replicacao_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
