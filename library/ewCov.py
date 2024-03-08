import numpy as np

def ewCovar(input, lambda_=0.97):
    ror = input.values
    ror_mean = np.mean(ror, axis=0)
    dev = ror - ror_mean
    times = dev.shape[0]
    weights = np.zeros(times)
    
    for i in range(times):
        weights[times - i - 1]  = (1 - lambda_) * lambda_**i
    
    weights_mat = np.diag(weights/sum(weights))

    cov = np.transpose(dev) @ weights_mat @ dev
    return cov
