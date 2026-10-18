"""Teste do GroupTargetEncoder. Uso: .venv/bin/python src/test_encoder.py"""
import numpy as np
import pandas as pd

from modelos import GroupTargetEncoder

# Dez acidentes com um óbito e um ileso cada, e o acidente 1 com quatro óbitos e
# uma marca só dele. As linhas do acidente 1 ficam intercaladas para que um
# KFold comum as espalhe por dobras diferentes.
linhas = [(g, f"m{g % 3}", y) for g in range(2, 12) for y in (1, 0)]
for i in range(4):
    linhas.insert(i * 6, (1, "exclusiva", 1))
X = pd.DataFrame(linhas, columns=["id", "marca", "y"])

out = GroupTargetEncoder(["marca"]).fit_transform(X[["marca", "id"]], X["y"])
# Sem o próprio acidente, a marca não aparece no treino e recebe o prior, 0,5.
assert np.allclose(out.loc[X["id"] == 1, "marca"], 0.5), out.loc[X["id"] == 1, "marca"]
print("ok")
