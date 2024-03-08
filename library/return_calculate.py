import numpy as np
import pandas as pd

def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars = [var for var in prices.columns if var != dateColumn]
    if not vars:
        raise ValueError(f"{dateColumn} not in DataFrame: {prices.columns}")

    p = np.matrix(prices[vars])
    n, m = p.shape
    p2 = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in ('LOG','DISCRETE')")

    dates = prices[dateColumn][1:n].reset_index(drop=True)
    data = {vars[i]: p2[:, i].flatten() for i in range(m)}
    data[dateColumn] = dates

    out = pd.DataFrame(data)
    return out

