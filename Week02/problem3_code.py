import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Function to fit ARIMA model and return AIC and BIC
def fit_arima(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.aic, model_fit.bic

# Load the data
file_path = '/Users/heyahe/FINTECH-545/Week02/problem3.csv'
data = pd.read_csv(file_path)

# AR models (ARIMA with p=1-3, d=0, q=0)
ar_results = {p: fit_arima(data['x'], (p, 0, 0)) for p in range(1, 4)}

# MA models (ARIMA with p=0, d=0, q=1-3)
ma_results = {q: fit_arima(data['x'], (0, 0, q)) for q in range(1, 4)}

# Print the AIC and BIC values
print("AIC and BIC values for AR models:")
for p, (aic, bic) in ar_results.items():
    print(f"AR({p}): AIC = {aic}, BIC = {bic}")

print("\nAIC and BIC values for MA models:")
for q, (aic, bic) in ma_results.items():
    print(f"MA({q}): AIC = {aic}, BIC = {bic}")
