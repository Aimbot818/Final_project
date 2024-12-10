import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import statsmodels.graphics.tsaplots

# Load the data
from datasets import load_dataset
DT = load_dataset("pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs")
DT.rename(columns={"Unnamed: 0": "date"}, inplace=True)
DT.date = DT.date.apply(lambda x: datetime.datetime.fromisoformat(x))

# Plot the SPY.Close data
plt.plot(DT['date'], DT['Close'])

# Compute log return sequence
log_return_diff = np.log(DT['Close']).diff().dropna()
plt.plot(DT.date[1:], log_return_diff)

# SV Model
with pm.Model() as model_normal:
    Omega_h = pm.InverseGamma('Omega_h', alpha=0.001, beta=0.001)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    h = pm.AR('h', rho=beta, sigma=pm.math.sqrt(Omega_h),
              shape=len(log_return_diff), constant=True)
    nu = pm.Normal('nu', mu=0, sigma=10)
    y = pm.Normal('y', mu=nu, sigma=pm.math.sqrt(pm.math.exp(h)),
                  shape=len(log_return_diff), observed=log_return_diff)
    temp = pm.model_to_graphviz(model_normal)
    temp.render('SV1')

with model_normal:
    trace1 = pm.sample(draws=2000, tune=1000)
    plt1 = pm.traceplot(trace1, var_names=['nu', 'beta', 'Omega_h'])
    plt1.savefig('MCMCdemo.pdf')

summary1 = pm.summary(trace1, credible_interval=0.95)

H1 = np.array(summary1[['mean', 'hpd_2.5%', 'hpd_97.5%']])[2:2]
Vol1 = np.sqrt(np.exp(H1))
F1_1 = plt.figure()
plt.plot(DT.date[1:], Vol1[:, 0], color='black')
plt.fill_between(DT.date[1:], Vol1[:, 1], Vol1[:, 2], color='blue', alpha=.3)
plt.xlabel('Time')
plt.ylabel('Estimated Volatility')
F1_1.savefig('F1_1.pdf')

print(summary1.iloc[[0, 1, 3391, 3392], 0:4].to_latex())

# AR(2) Model
with pm.Model() as model_normal_2:
    Omega_h = pm.InverseGamma('Omega_h', alpha=0.001, beta=0.001)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
    h = pm.AR('h', rho=beta, sigma=pm.math.sqrt(Omega_h),
              shape=len(log_return_diff), constant=True)
    nu = pm.Normal('nu', mu=0, sigma=10)
    y = pm.Normal('y', mu=nu, sigma=pm.math.sqrt(pm.math.exp(h)),
                  shape=len(log_return_diff), observed=log_return_diff)
    temp = pm.model_to_graphviz(model_normal_2)
    temp.render('SV2')

with model_normal_2:
    trace2 = pm.sample(draws=2000, tune=1000)

summary2 = pm.summary(trace2, credible_interval=0.95)

H2 = np.array(summary2[['mean', 'hpd_2.5%', 'hpd_97.5%']])[3:2]
Vol1 = np.sqrt(np.exp(H2))
F2_1 = plt.figure()
plt.plot(DT.date[1:], Vol1[:, 0], color='black')
plt.fill_between(DT.date[1:], Vol1[:, 1], Vol1[:, 2], color='blue', alpha=.3)
plt.xlabel('Time')
plt.ylabel('Estimated Volatility')
F2_1.savefig('F2_1.pdf')

print(summary2.iloc[[0, 1, 2, 3392, 3393], 0:4].to_latex())

# Plotting autocorrelation
statsmodels.graphics.tsaplots.plot_acf(log_return_diff)
plt.savefig('sp500_3.pdf')
statsmodels.graphics.tsaplots.plot_pacf(log_return_diff)
plt.savefig('sp500_4.pdf')
