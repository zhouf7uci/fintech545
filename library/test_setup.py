import pandas as pd
import numpy as np
from scipy.stats import norm, t, spearmanr
from missing_cov import missing_cov
from ewCov import ewCovar
from psd import near_psd, higham_nearestPSD
from chol_psd import chol_psd
from simulate import simulateNormal, simulate_pca
from return_calculate import return_calculate
from fitted_model import fit_normal, fit_general_t, fit_regression_t
from var import var_normal, var_t, var_simulation
from es import es_normal, es_t, es_simulation
from copula import simulate_copula

# Function to compare the computed result with the expected result
def compare_results(num, computed, expected, decimal=7):
    try:
        np.testing.assert_array_almost_equal(computed, expected, decimal=decimal)
        print(f"Test{num} Passed: Computed result matches the expected result.")
    except AssertionError as e:
        print(f"Test{num} Failed: {e}")

#Test 1 - missing covariance calculations
x1 = pd.read_csv("library/data/test1.csv")

#1.1 Skip Missing rows - Covariance
num = 1.1
cout1_1 = missing_cov(x1,skipMiss=True)
testout1_1 = pd.read_csv("library/data/testout_1.1.csv", header=0).values
compare_results(num, cout1_1, testout1_1)

#1.2 Skip Missing rows - Correlation
num = 1.2
cout1_2 = missing_cov(x1,skipMiss=True,fun=np.corrcoef)
testout1_2 = pd.read_csv("library/data/testout_1.2.csv", header=0).values
compare_results(num, cout1_2, testout1_2)

#1.3 Pairwise - Covariance
num = 1.3
cout1_3 = missing_cov(x1,skipMiss=False)
testout1_3 = pd.read_csv("library/data/testout_1.3.csv", header=0).values
compare_results(num, cout1_3, testout1_3)

#1.4 Pairwise - Correlation
num = 1.4
cout1_4 = missing_cov(x1,skipMiss=False,fun=np.corrcoef)
testout1_4 = pd.read_csv("library/data/testout_1.4.csv", header=0).values
compare_results(num, cout1_4, testout1_4)

#Test 2 - EW Covariance
x2 = pd.read_csv("library/data/test2.csv")

#2.1 EW Covariance λ=0.97
num = 2.1
cout2_1 = ewCovar(x2,0.97)
testout2_1 = pd.read_csv("library/data/testout_2.1.csv", header=0).values
compare_results(num, cout2_1, testout2_1)

#2.2 EW Correlation λ=0.94
num = 2.2
cout = ewCovar(x2,0.94)
sd = 1 / np.sqrt(np.diag(cout))
cout2_2 = np.diag(sd) @ cout @ np.diag(sd)
testout2_2 = pd.read_csv("library/data/testout_2.2.csv", header=0).values
compare_results(num, cout2_2, testout2_2)

#2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
num = 2.3
cout = ewCovar(x2,0.97)
sd1 = np.sqrt(np.diag(cout))
cout = ewCovar(x2,0.94)
sd = 1 / np.sqrt(np.diag(cout))
cout2_3 = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
testout2_3 = pd.read_csv("library/data/testout_2.3.csv", header=0).values
compare_results(num, cout2_3, testout2_3)

#Test 3 - non-psd matrices
#3.1 near_psd covariance
num = 3.1
cin = pd.read_csv("library/data/testout_1.3.csv")
cout3_1 = near_psd(cin)
testout3_1 = pd.read_csv("library/data/testout_3.1.csv", header=0).values
compare_results(num, cout3_1, testout3_1)

#3.2 near_psd Correlation
num = 3.2
cin = pd.read_csv("library/data/testout_1.4.csv")
cout3_2 = near_psd(cin)
testout3_2 = pd.read_csv("library/data/testout_3.2.csv", header=0).values
compare_results(num, cout3_2, testout3_2)

#3.3 Higham covariance
num = 3.3
cin = pd.read_csv("library/data/testout_1.3.csv")
cout3_3 = higham_nearestPSD(cin)
testout3_3 = pd.read_csv("library/data/testout_3.3.csv", header=0).values
compare_results(num, cout3_3, testout3_3)

#3.4 Higham Correlation
num = 3.4
cin = pd.read_csv("library/data/testout_1.4.csv")
cout3_4 = higham_nearestPSD(cin)
testout3_4 = pd.read_csv("library/data/testout_3.4.csv", header=0).values
compare_results(num, cout3_4, testout3_4)

#4 cholesky factorization
num = 4
cin = pd.read_csv("library/data/testout_3.1.csv").values  # the input is .values
cout4 = chol_psd(cin)
testout4_1 = pd.read_csv("library/data/testout_4.1.csv", header=0).values
compare_results(num, cout4, testout4_1)

#5 Normal Simulation
#5.1 PD Input
num = 5.1
cin = pd.read_csv("library/data/test5_1.csv")
simulated_data = simulateNormal(100000, cin)
cout5_1 = np.cov(simulated_data)
testout5_1 = pd.read_csv("library/data/testout_5.1.csv", header=0).values
compare_results(num, cout5_1, testout5_1, 3)

# 5.2 PSD Input
num = 5.2
cin = pd.read_csv("library/data/test5_2.csv")
simulated_data = simulateNormal(100000, cin)
cout5_2 = np.cov(simulated_data)
testout5_2 = pd.read_csv("library/data/testout_5.2.csv", header=0).values
compare_results(num, cout5_2, testout5_2, 3)

# 5.3 nonPSD Input, near_psd fix
num = 5.3
cin = pd.read_csv("library/data/test5_3.csv")
simulated_data = simulateNormal(100000, cin, fixMethod=near_psd)
cout5_3 = np.cov(simulated_data)
testout5_3 = pd.read_csv("library/data/testout_5.3.csv", header=0).values
compare_results(num, cout5_3, testout5_3, 3)

# 5.4 nonPSD Input Higham Fix
num = 5.4
cin = pd.read_csv("library/data/test5_3.csv")
simulated_data = simulateNormal(100000, cin, fixMethod=higham_nearestPSD)
cout5_4 = np.cov(simulated_data)
testout5_4 = pd.read_csv("library/data/testout_5.4.csv", header=0).values
compare_results(num, cout5_4, testout5_4, 3)

# 5.5 PSD Input - PCA Simulation
num = 5.5
cin = pd.read_csv("library/data/test5_2.csv")
simulated_data = simulate_pca(cin,100000,pctExp=.99)
cout5_5 = np.cov(simulated_data)
testout5_5 = pd.read_csv("library/data/testout_5.5.csv", header=0).values
compare_results(num, cout5_5, testout5_5, 3)

# Test 6
# 6.1 Arithmetic returns
num = 6.1
prices = pd.read_csv("library/data/test6.csv")
rout6_1 = return_calculate(prices,dateColumn="Date")
testout6_1 = pd.read_csv("library/data/test6_1.csv")
computed_array = rout6_1.drop(columns=['Date']).to_numpy()
expected_array = testout6_1.drop(columns=['Date']).to_numpy()
compare_results(num, computed_array, expected_array)

# 6.2 Log returns
num = 6.2
prices = pd.read_csv("library/data/test6.csv")
rout6_2 = return_calculate(prices,method="LOG", dateColumn="Date")
testout6_2 = pd.read_csv("library/data/test6_2.csv")
computed_array = rout6_2.drop(columns=['Date']).to_numpy()
expected_array = testout6_2.drop(columns=['Date']).to_numpy()
compare_results(num, computed_array, expected_array)

# Test 7
# 7.1 Fit Normal Distribution
num = 7.1
cin = pd.read_csv("library/data/test7_1.csv")
mu, sigma = fit_normal(cin)
output_df1 = pd.DataFrame({'mu': [mu], 'sigma': [sigma]})
testout7_1 = pd.read_csv("library/data/testout7_1.csv")
compare_results(num, output_df1, testout7_1, 3)

# 7.2 Fit TDist
num = 7.2
cin = pd.read_csv("library/data/test7_2.csv")
mu, sigma, nu = fit_general_t(cin)
output_df2 = pd.DataFrame({'mu': [mu], 'sigma': [sigma], 'nu': [nu]})
testout7_2 = pd.read_csv("library/data/testout7_2.csv")
compare_results(num, output_df2, testout7_2, 3)

# 7.3 Fit T Regression
num = 7.3
cin = pd.read_csv("library/data/test7_3.csv")
output_df3= fit_regression_t(cin)
testout7_3 = pd.read_csv("library/data/testout7_3.csv")
compare_results(num, output_df3, testout7_3, 3)

# Test 8
# Test 8.1 VaR Normal
num = 8.1
cin = pd.read_csv("library/data/test7_1.csv")
output_df1= var_normal(cin)
testout8_1 = pd.read_csv("library/data/testout8_1.csv")
compare_results(num, output_df1, testout8_1, 3)

# Test 8.2 VaR TDist
num = 8.2
cin = pd.read_csv("library/data/test7_2.csv")
output_df2= var_t(cin)
testout8_2 = pd.read_csv("library/data/testout8_2.csv")
compare_results(num, output_df2, testout8_2, 3)

# Test 8.3 VaR Simulation
num = 8.3
cin = pd.read_csv("library/data/test7_2.csv")
output_df3= var_simulation(cin)
testout8_3 = pd.read_csv("library/data/testout8_3.csv")
compare_results(num, output_df3, testout8_3, 2)

# Test 8.4 ES Normal
num = 8.4
cin = pd.read_csv("library/data/test7_1.csv")
output_df4= es_normal(cin)
testout8_4 = pd.read_csv("library/data/testout8_4.csv")
compare_results(num, output_df4, testout8_4, 3)

# Test 8.5 ES TDist
num = 8.5
cin = pd.read_csv("library/data/test7_2.csv")
output_df5= es_t(cin)
testout8_5 = pd.read_csv("library/data/testout8_5.csv")
compare_results(num, output_df5, testout8_5, 3)

# Test 8.6 ES Simulation
num = 8.6
cin = pd.read_csv("library/data/test7_2.csv")
output_df6= es_simulation(cin)
testout8_6 = pd.read_csv("library/data/testout8_6.csv")
compare_results(num, output_df6, testout8_6, 3)

# Test 9
# 9.1
num = 9.1
returns = pd.read_csv("library/data/test9_1_returns.csv")
portfolio = pd.read_csv("library/data/test9_1_portfolio.csv")
risk = simulate_copula(portfolio, returns)
results = risk.drop(columns=['Stock']).to_numpy()
testout9_1 = pd.read_csv("library/data/testout9_1.csv").drop(columns=['Stock']).to_numpy()
compare_results(num, results, testout9_1, 0)



# cin = CSV.read("data/test9_1_returns.csv",DataFrame)
# prices = Dict{String,Float64}()
# prices["A"] = 20.0
# prices["B"] = 30

# models = Dict{String,FittedModel}()
# models["A"] = fit_normal(cin.A)
# models["B"] = fit_general_t(cin.B)

# nSim = 100000

# U = [models["A"].u models["B"].u]
# spcor = corspearman(U)
# uSim = simulate_pca(spcor,nSim)
# uSim = cdf.(Normal(),uSim)

# simRet = DataFrame(:A=>models["A"].eval(uSim[:,1]), :B=>models["B"].eval(uSim[:,2]))

# portfolio = DataFrame(:Stock=>["A","B"], :currentValue=>[2000.0, 3000.0])
# iteration = [i for i in 1:nSim]
# values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

# nv = size(values,1)
# pnl = Vector{Float64}(undef,nv)
# simulatedValue = copy(pnl)
# for i in 1:nv
#     simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
#     pnl[i] = simulatedValue[i] - values.currentValue[i]
# end

# values[!,:pnl] = pnl
# values[!,:simulatedValue] = simulatedValue

# risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

# CSV.write("data/testout9_1.csv",risk)