#problem 1
import math
import numpy as np
import sys
import pandas as pd
from numpy.linalg import eigh
import itertools
from scipy.stats import norm, t
import statsmodels.api as sm
from scipy.optimize import fsolve, minimize
import inspect

''' Calculation of covariance matrix '''
# Define a function to calculate exponential weights
def populateWeights(x, w, cw, λ):
    n = len(x)
    tw = 0
    # Use for loop to calculate the weight for each stock, get the total weights and cumulative weights for each stock
    for i in range(n):
        individual_w = (1-λ)*pow(λ,i)
        w.append(individual_w)
        tw += individual_w
        cw.append(tw)
    # Calculate normalized weights and normalized cumulative weights for each stock
    for i in range(n):
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw

# Calculate the exponentially weighted covariance matrix
def exwCov(data, weights_vector):
    # Get the stock names listed in the file, delete the first item, since it is the column of dates
    stock_names = list(data.columns)
    # Set up an empty matrix, and transform it into a pandas Dataframe
    mat = np.empty((len(stock_names),len(stock_names)))
    w_cov_mat = pd.DataFrame(mat, columns = stock_names, index = stock_names)
    # Calculate variances and covariances
    for i in stock_names:
        for j in stock_names:
            # Get data of stock i and data of stock j respectively
            i_data = data.loc[:,i]
            j_data = data.loc[:,j]
            # Calculate means of data of stock i and data of stock j
            i_mean = i_data.mean()
            j_mean = j_data.mean()
            # Ensure i_data, j_data, and weights_vector all have the same number of items
            assert len(i_data) == len(j_data) == len(weights_vector)
            # set up sum for calculation of variance and covariance, and a for loop for that
            sum = 0
            for z in range(len(data)):                
                part = weights_vector[z] * (i_data[z] - i_mean) * (j_data[z] - j_mean)
                sum += part
            # store the derived variance into the matrix
            w_cov_mat.loc[i,j] = sum
    return w_cov_mat


'''Fix non-psd matrices'''
# Cholesky factorization for psd matrices
def chol_psd_forpsd(root,a):
    n = a.shape
    # Initialize root matrix with zeros
    root = np.zeros(n)
    for j in range(n[0]):
        sum = 0
        # If not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            sum = np.dot(root[j,:(j-1)], root[j,:(j-1)])
        # Diagonal elements
        temp = a[j,j] - sum
        if temp <= 0 and temp >= -1e-8:
            temp = 0
        root[j,j] = math.sqrt(temp)
        # Check for zero eigenvalues; set columns to zero if we have one(s)
        if root[j,j] == 0:
            root[j,(j+1):] = 0
        else:
            ir = 1/root[j,j]
            for i in range((j+1),n[0]):
                sum = np.dot(root[i,:(j-1)], root[j,:(j-1)])
                root[i,j] = (a[i,j] - s) * ir
    return root

# Cholesky factorization for pd matrices                
def chol_psd_forpd(root,a):
    n = a.shape
    # Initialize root matrix with zeros
    root = np.zeros(n)
    for j in range(n[0]):
        sum = 0
        # If not on the first column, calculate the dot product of the preceeding row values
        if j > 1:
            sum = np.dot(root[j,:(j-1)], root[j,:(j-1)])
        # Diagonal elements
        temp = a[j,j] - sum
        root[j,j] = math.sqrt(temp)
        ir = 1/root[j,j]
        # Update off diagonal rows of the column
        for i in range((j+1),n[0]):
            sum = np.dot(root[i,:(j-1)], root[j,:(j-1)])
            root[i,j] = (a[i,j] - sum) * ir
    return root            

# Calculate PSD via near_PSD
def near_PSD(matrix, epsilon=0.0):
    # If the matrix is a correlation matrix - if all numbers on the diagonal are one
    matrix_diag = np.diag(matrix)
    for i in matrix_diag:
        assert i == 1
    # Calculate the eigenvalues and eigenvectors
    e_val, e_vec = eigh(matrix)
    # Sort eigenvalues and corresponding eigenvectors in a descending order
    index = np.argsort(-1 * e_val)
    d_e_val = e_val[index]
    d_e_vec = e_vec[:,index]
    # Set eigenvalues that are smaller than epsilon to epsilon
    d_e_val[d_e_val < epsilon] = epsilon
    # Construct the scaling diagonal matrix, calculating t(s) and store them into the list called t_vec
    t_vec = []
    for i in range(len(d_e_val)):
        sum_t = 0
        for j in range(len(d_e_val)):
            t = pow(d_e_vec[i][j],2) * d_e_val[j]
            sum_t += t
        t_i = 1 / sum_t
        t_vec.append(t_i)
    # Construct the resulting near_PSD matrix
    B_matrix = np.diag(np.sqrt(t_vec)) @ d_e_vec @ np.diag(np.sqrt(d_e_val))
    B_matrix_transpose = B_matrix.transpose()
    C_prime_matrix = B_matrix @ B_matrix_transpose
    # If eigenvalues are all non-negative now (assuming all significantly small eigenvalues are zero, the tolerance level here is set to be -1e-8)
    result_vals, result_vecs = eigh(C_prime_matrix)
    neg_result_vals = result_vals[result_vals < 0]
    if neg_result_vals.any() < -1e-8:
        print("There are still significantly negative eigenvalues, recommend to run the function again over the result until a PSD is generated")
    
    return C_prime_matrix


# Calculate PSD via Higham's method
def Higham(A, tolerance=1e-8):
    # set up delta S, Y, and gamma
    delta_s = np.full(A.shape,0)
    Y = A.copy()
    gamma_last = sys.float_info.max
    gamma_now = 0
    # Start the actual iteration
    for i in itertools.count(start=1):        
        R = Y - delta_s
        # Conduct the second projection of Higham's method over R
        Rval, Rvec = eigh(R)
        Rval[Rval < 0] = 0
        Rvec_transpose = Rvec.transpose()
        X = Rvec @ np.diag(Rval) @ Rvec_transpose
        delta_s = X - R
        # Conduct the first projection of Higham's method over X
        size_X = X.shape
        for i in range(size_X[0]):
            for j in range(size_X[1]):
                if i == j:
                    Y[i][j] = 1
                else:
                    Y[i][j] = X[i][j]
        difference_mat = Y - A
        gamma_now = F_Norm(difference_mat)
        # Get eigenvalues and eigenvectors of updated Y
        Yval, Yvec = eigh(Y)
        # Breaking conditions
        if np.amin(Yval) > -1*tolerance:
            break
        else:
            gamma_last = gamma_now
    
    return Y


'''Simulations method(PCA, etc.)'''
# PCA to simulate the system through defining a new function
def simulate_PCA(a, nsim, percent_explained=1):
    # Calculate the eigenvalues and eigenvectors of derived matrix, and sort eigenvalues from largest to smallest
    e_val, e_vec = eigh(a)
    sort_index = np.argsort(-1 * e_val)
    d_sorted_e_val = e_val[sort_index]
    d_sorted_e_vec = e_vec[:,sort_index]
    # All negative eigenvalues derived are zero, since they are effectively zero (larger than -1e-8)
    assert np.amin(d_sorted_e_val) > -1e-8
    d_sorted_e_val[d_sorted_e_val<0] = 0
    # Sum of all eigenvalues
    e_sum = sum(d_sorted_e_val)
    # Choose a certain number of eigenvalues from the descending list of all eigenvalues so that the system explains the same percent of variance as the level inputed as parameter "percent_explained"
    total_percent = []
    sum_percent = 0
    for i in range(len(d_sorted_e_val)):
        each_percent = d_sorted_e_val[i] / e_sum
        sum_percent += each_percent
        total_percent.append(sum_percent)
    total_percent_np = np.array(total_percent)
    diff = total_percent_np - percent_explained
    abs_diff = abs(diff)
    index = np.where(abs_diff==abs_diff.min())
    # Update eigenvalues and eigenvectors with the list of indices we generate above
    upd_e_val = d_sorted_e_val[:(index[0][0]+1)]
    upd_e_vec = d_sorted_e_vec[:,:(index[0][0]+1)]
    # Construct the matrix for the simulating process
    B = upd_e_vec @ np.diag(np.sqrt(upd_e_val))
    r = np.random.randn(len(upd_e_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t

# Direct simulation
def direct_simulate(a, nsim):
    # Get eigenvalues and eigenvectors of the input matrix
    val, vec = eigh(a)
    sort_index = np.argsort(-1 * val)
    d_sorted_val = val[sort_index]
    d_sorted_vec = vec[:,sort_index]
    # If all eigenvalues are non-negative or negative but effectively zero, and set all effectively-zero eigenvalues to zero
    assert np.amin(d_sorted_val) > -1e-8
    d_sorted_val[d_sorted_val<0] = 0
    
    B = d_sorted_vec @ np.diag(np.sqrt(d_sorted_val))
    r = np.random.randn(len(d_sorted_val),nsim)
    
    result = B @ r
    result_t = np.transpose(result)
    
    return result_t


'''VaR'''
# Assume no distribution
def cal_VaR(x,alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    return -VaR

# Another way without assuming any distribution
def comp_VaR(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)

# Basic distributions (only normal, t, and AR(1) are available in this function)
def VaR_bas(data, alpha=0.05, dist="normal", n=10000):
    # Centralize data
    data = data - data.mean()
    if dist=="normal":
        fit_result = norm.fit(data)
        return -norm.ppf(alpha, loc=fit_result[0], scale=fit_result[1])
    elif dist=="t":
        fit_result = t.fit(data)
        return -t.ppf(alpha, df=fit_result[0], loc=fit_result[1], scale=fit_result[2])
    elif dist=="ar1":
        mod = sm.tsa.ARIMA(data, order=(1, 0, 0))
        fit_result = mod.fit()
        summary = fit_result.summary()
        m = float(summary.tables[1].data[1][1])
        a1 = float(summary.tables[1].data[2][1])
        s = math.sqrt(float(summary.tables[1].data[3][1]))
        out = np.zeros(n)
        sim = np.random.normal(size=n)
        data_last = data.iloc[-1] - m
        for i in range(n):
            out[i] = a1 * data_last + sim[i] * s + m
        return comp_VaR(out, mean=out.mean())
    else:
        return "Invalid distribution in this method."

# EWMA VaR for portfolios (check the order of data, if it is from farthest to nearest, this is correct; if not, plz modify the code or reverse the order to "farthest and nearest"; make sure that there should not be a date column in returns)
def del_norm_VaR(current_prices, holdings, returns, lamda=0.94, alpha=0.05):
    # demean returns
    returns -= returns.mean()
    w = []
    cw = []
    PV = 0
    delta = np.zeros(len(holdings))
    populateWeights(returns, w, cw, lamda)
    w = w[::-1]
    cov = exwCov(returns, w)
    for i in range(len(holdings)):
        temp_holding = holdings.iloc[i,-1] 
        value = temp_holding * current_prices[i]
        PV += value
        delta[i] = value
    delta = delta / PV
    fac = math.sqrt(np.transpose(delta) @ cov @ delta)
    VaR = -PV * norm.ppf(alpha, loc=0, scale=1) * fac
    return VaR

# Historic VaR (when used, check how returns are derived; log returns are fine; arithmetic returns should be changed the way of calculation simulated prices; also, there should not be a date column in returns)
def hist_VaR(current_prices, holdings, returns, alpha=0.05):
    # centralize returns
    returns -= returns.mean()
    PV = 0
    for i in range(len(holdings)):
        value = holdings.iloc[i,-1] * current_prices[i]
        PV += value
    sim_prices = (np.exp(returns)) * np.transpose(current_prices)
    port_values = np.dot(sim_prices, holdings.iloc[:,-1])
    port_values_sorted = np.sort(port_values)
    index = np.floor(alpha*len(returns))
    VaR = PV - port_values_sorted[int(index-1)]
    return VaR

# Monte Carlo normal VaR (note that when used, check how returns are derived; if they are log returns, you are fine; if they are arithmetic returns, change the way you calculate simulated prices)
def MC_VaR(current_prices, holdings, returns, n=10000, alpha=0.05):
    # demean returns
    returns -= returns.mean()
    PV = 0
    for i in range(len(holdings)):
        value = holdings.iloc[i,-1] * current_prices[i]
        PV += value
    sim_returns = np.random.multivariate_normal(returns.mean(), returns.cov(), (1,len(holdings),n))
    sim_returns = np.transpose(sim_returns)
    sim_prices = (np.exp(returns)) * np.transpose(current_prices)
    port_values = np.dot(sim_prices, holdings.iloc[:,-1])
    port_values_sorted = np.sort(port_values)
    index = np.floor(alpha*n)
    VaR = PV - port_values_sorted[int(index-1)]
    return VaR


'''ES calculation'''
# ES calculation of individual data
def cal_ES(x,alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = math.ceil(n)
    idn = math.floor(n)
    VaR = (xs[iup] + xs[idn]) / 2
    ES = xs[0:idn].mean()
    return VaR,ES