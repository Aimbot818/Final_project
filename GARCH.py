import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf, q_stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the data
from datasets import load_dataset
ds_1 = load_dataset("pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs")
data = pd.DataFrame(ds_1)
x = data['Close']

# Log returns
r = np.diff(np.log(x))

# Stationarity test (ADF Test)
adf_result = adfuller(r)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# White noise test (Box-Ljung Test)
acf_values = acf(r, fft=False)
q_stats, p_values = q_stat(acf_values[1:], len(r))
print("Box-Ljung Test p-value:", p_values)

# ACF and PACF plots
plot_acf(r, lags=40, title="ACF of Log Returns")
plot_pacf(r, lags=40, title="PACF of Log Returns")
plt.show()

# ARIMA Model Order Selection (using auto_arima equivalent)
from pmdarima import auto_arima
arima_order = auto_arima(r, seasonal=False, trace=True).order
print("Selected ARIMA Order:", arima_order)

# GARCH Model
garch_model = arch_model(r, vol='Garch', p=1, q=1, mean='AR', lags=1, dist='normal')
garch_fit = garch_model.fit()
print(garch_fit.summary())

# Residual Analysis
residuals = garch_fit.resid / garch_fit.conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title("Standardized Residuals")
plt.show()

# Volatility Plot
volatility = garch_fit.conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(volatility, label="Estimated Volatility")
plt.axhline(np.std(r), color='red', linestyle='--', label="Sample Standard Deviation")
plt.legend()
plt.title("GARCH(1,1) Volatility")
plt.show()

# Additional Models
# EGARCH Model
egarch_model = arch_model(r, vol='EGarch', p=1, q=1, mean='AR', lags=1, dist='normal')
egarch_fit = egarch_model.fit()
print(egarch_fit.summary())

# GARCH Forecast
forecast = garch_fit.forecast(horizon=1)
forecast_mean = forecast.mean.iloc[-1].values
forecast_volatility = forecast.variance.iloc[-1].values
print("GARCH Forecast Mean:", forecast_mean)
print("GARCH Forecast Volatility:", forecast_volatility)

# Handling COVID Influence
ds_2 = pd.read_csv("bing_covid-19_data.csv")
data2 = pd.DataFrame(ds_2)
x2 = data2['confirmed']
r2 = np.diff(np.log(x2))
covid = data2['log_diff_positive_2'].iloc[1:]

# Stationarity Test for the New Series
adf_result2 = adfuller(r2)
print("ADF Statistic (COVID Data):", adf_result2[0])
print("p-value (COVID Data):", adf_result2[1])

# GARCH with External Regressors
garch_ext_model = arch_model(r2, vol='EGarch', p=1, q=1, mean='AR', lags=1, dist='normal', x=covid)
garch_ext_fit = garch_ext_model.fit()
print(garch_ext_fit.summary())
