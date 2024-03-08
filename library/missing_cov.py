import numpy as np
import pandas as pd

def missing_cov(x, skipMiss=True, fun=np.cov):
    # Convert input to a DataFrame for easier missing values handling
    x_df = pd.DataFrame(x)
    n, m = x_df.shape
    nMiss = x_df.isnull().sum()

    # Nothing missing, calculate it
    if nMiss.sum() == 0:
        return fun(x_df.values, rowvar=False)

    idxMissing = [x_df.index[x_df.iloc[:, col].isnull()] for col in range(m)]
    
    if skipMiss:
        # Skipping Missing, get all the rows which have values and calculate the covariance
        rows = set(range(n))
        for c in range(m):
            rows -= set(idxMissing[c])
        rows = sorted(rows)
        return fun(x_df.iloc[rows, :].dropna().values, rowvar=False)
    else:
        # Pairwise, for each cell, calculate the covariance.
        out = np.empty((m, m))
        for i in range(m):
            for j in range(i + 1):
                rows = set(range(n))
                for c in (i, j):
                    rows -= set(idxMissing[c])
                rows = sorted(rows)
                subset = x_df.iloc[rows, [i, j]].dropna()
                if len(subset) > 1:
                    out[i, j] = fun(subset.values, rowvar=False)[0, 1]
                else:
                    out[i, j] = np.nan
                if i != j:
                    out[j, i] = out[i, j]
        return out
