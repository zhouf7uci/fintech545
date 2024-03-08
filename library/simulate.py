import numpy as np
from chol_psd import chol_psd
from psd import near_psd, higham_nearestPSD
from numpy.linalg import eigh

def simulateNormal(N, cov, mean=None, seed_val=1234, fixMethod=near_psd):
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    out = np.zeros((N,n))

    # Set the mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Generate needed random standard normals
    np.random.seed(seed_val)

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    if min(eigenvalues) < 0:
        cov = fixMethod(cov)
        l = chol_psd(cov)
    else:
        l = chol_psd(cov.values)

    rand_normals = np.random.normal(0.0, 1.0, size=(N, n))

    out = np.dot(rand_normals, l.T) + mean

    return out.T


def simulate_pca(a, nsim, pctExp=1, mean=None, seed_val=1234):
    m, n = a.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")
    out = np.zeros((nsim,n))

    # Set the mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    indices = np.argsort(vals)[::-1]
    vals = vals[indices]
    vecs = vecs[:, indices]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        for i in posv:
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]
    B = vecs @ np.diag(np.sqrt(vals))

    np.random.seed(seed_val)
    rand_normals = np.random.normal(0.0, 1.0, size=(nsim, len(posv)))
    out = np.dot(rand_normals, B.T) + mean
    return out.T
