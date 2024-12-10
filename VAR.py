import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VARResults

# Download data
def get_log_diff(symbol, start_date="2000-01-01", end_date="2020-01-01"):
    # Fetch stock data using yfinance
    data = yf.download(symbol, start=start_date, end=end_date)
    close = data['Close']
    log_diff = np.diff(np.log(close))
    return log_diff

# Get data for three stocks
d1 = get_log_diff("SPY")
d2 = get_log_diff("AAPL")
d3 = get_log_diff("C")

# Plot log-difference data
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(d1, label="S&P 500")
plt.title("S&P 500")
plt.subplot(3, 1, 2)
plt.plot(d2, label="Apple")
plt.title("Apple")
plt.subplot(3, 1, 3)
plt.plot(d3, label="Citibank")
plt.title("Citibank")
plt.tight_layout()
plt.show()

# Combine into a multivariate time series
x = np.column_stack((d1, d2, d3))
x_df = pd.DataFrame(x, columns=["SPY", "AAPL", "C"])

# Granger causality tests
def granger_test(x, y, max_lag=10):
    result = grangercausalitytests(np.column_stack((x, y)), max_lag, verbose=False)
    p_values = [round(result[i + 1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
    return p_values

print("Granger Test SPY -> AAPL:", granger_test(d1, d2))
print("Granger Test SPY -> C:", granger_test(d1, d3))
print("Granger Test AAPL -> C:", granger_test(d2, d3))

# VAR model order selection
model = VAR(x_df)
selected_lag = model.select_order(maxlags=10).selected_orders['aic']
print(f"Selected lag order: {selected_lag}")

# Fit VAR model
var_model = model.fit(selected_lag)
print(var_model.summary())

# Leave-one-out validation
test_indices = [3023, 274, 1568, 1079, 2365, 1129, 1887, 666, 2448, 3217, 969, 748]
s = []
for i in test_indices:
    fit = model.fit(9)
    prediction = fit.forecast(y=x[:i, :], steps=1)[0, 0]
    true_value = d1[i + 1]
    residuals = fit.resid[:, 0]
    density = np.histogram(residuals, bins=100, density=True)
    gap = abs(prediction - true_value)
    closest_idx = np.searchsorted(density[1], gap)
    lambda_val = (gap - density[1][closest_idx - 1]) / (density[1][closest_idx] - density[1][closest_idx - 1])
    score = np.log(density[0][closest_idx - 1] * (1 - lambda_val) + density[0][closest_idx] * lambda_val)
    s.append(score)

print("Mean Score:", np.mean(s))

# Load COVID data and analyze its impact on S&P 500
covid_data = pd.read_csv("C:/Users/leoxy/PycharmProjects/Final_Project/GARCH/bing_covid-19_data.csv", index_col=0)
d1 = np.diff(np.log(covid_data.iloc[:, 0]))  # S&P 500
d2 = np.diff(np.log(covid_data.iloc[:, 1]))  # COVID data

# Granger causality test
granger_test_result = granger_test(d1, d2)
print("Granger Test COVID -> S&P 500:", granger_test_result)

# Fit VAR model
x_covid = np.column_stack((d1, d2))
var_covid_model = VAR(x_covid).fit(1)
print(var_covid_model.summary())
