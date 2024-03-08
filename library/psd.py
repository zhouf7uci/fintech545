import numpy as np

# define a function calculating PSD via near_PSD
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = np.matmul(np.matmul(invSD, out), invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.matmul(np.square(vecs), vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

def Frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result

# define a function calculating PSD via Higham's method
def higham_nearestPSD(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = Frobenius(Yk-Y0)
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk