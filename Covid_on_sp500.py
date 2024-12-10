import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import datetime
import pymc3 as pm

# Load SPY data
from datasets import load_dataset
SPY = load_dataset("pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs")
SPY.columns = ['date', 'SPY']
SPY.date = SPY.date.apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
SPY = SPY.drop(0)

# Load COVID data
COV = pd.read_csv('C:/Users/leoxy/PycharmProjects/Final_Project/GARCH/bing_covid-19_data.csv')
COV = COV.drop(columns='Unnamed: 0')
COV.date = COV.date.apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
COV = COV.sort_values(by='date')
COV['log_diff_positive'] = np.log(COV.positive).diff()
COV['log_diff_hospital'] = np.log(COV.hospitalizedCurrently).diff()
COV['log_diff_death'] = np.log(COV.death).diff()
COV = COV.fillna(0).reset_index(drop=True)

SPY['case'] = SPY.date.apply(lambda x: np.array(COV.loc[COV.date == x, ['positive']]).squeeze())
SPY['hospital'] = SPY.date.apply(lambda x: np.array(COV.loc[COV.date == x, ['hospitalizedCurrently']]).squeeze())

# Plot S&P 500 index
plt.plot(SPY.date, SPY.SPY * 10)
plt.xlabel('time')
plt.ylabel('S&P 500 index')
plt.savefig("SP_COVID_1.pdf")

# Plot COVID cases
plt.plot(SPY.date, SPY.case)
plt.xlabel('time')
plt.ylabel('COVID Cases')
plt.savefig("SP_COVID_2.pdf")

SPY['log_diff_positive'] = np.log(SPY.case).diff()
SPY['log_diff_SPY'] = np.log(SPY.SPY).diff()
SPY = SPY.dropna()

# Plot log differences
plt.plot(SPY.date, SPY.log_diff_SPY * 10)
plt.xlabel('time')
plt.ylabel('log difference of S&P 500 index')
plt.savefig("SP_COVID_3.pdf")

plt.plot(SPY.date, SPY.log_diff_positive)
plt.xlabel('time')
plt.ylabel('log difference of COVID case')
plt.savefig("SP_COVID_4.pdf")

SPY.to_csv('SPY_COV.csv')

SPY['log_diff_positive'] = SPY.date.apply(lambda x: np.array(COV.loc[COV.date == x, ['log_diff_positive']]).squeeze())
SPY['log_diff_hospital'] = SPY.date.apply(lambda x: np.array(COV.loc[COV.date == x, ['log_diff_hospital']]).squeeze())
SPY['log_diff_death'] = SPY.date.apply(lambda x: np.array(COV.loc[COV.date == x, ['log_diff_death']]).squeeze())

SPY["log_diff_positive_2"] = 0.9
SPY['log_diff_hospital_2'] = 0.0
SPY["log_diff_death_2"] = 0.0
temp = np.array([0, 0, 0])

for i in COV.date:
    if i not in set(SPY.date):
        temp = temp + COV.loc[COV.date == i, ["log_diff_positive", 'log_diff_hospital', "log_diff_death"]].to_numpy().squeeze()
    else:
        SPY.loc[SPY.date == i, ["log_diff_positive_2", 'log_diff_hospital_2', "log_diff_death_2"]] = temp + SPY.loc[SPY.date == i, ["log_diff_positive", 'log_diff_hospital', "log_diff_death"]].to_numpy().squeeze()
        temp[:] = 0

SPY.to_csv('SPY_USCOVID.csv')

# Bayesian modeling with pymc3
log_return_diff = np.log(SPY.SPY).diff()[1:]
log_positive_diff = SPY.log_diff_positive_2[1:]

with pm.Model() as model_T_mu:
    Omega_h = pm.InverseGamma('Omega_h', alpha=0.001, beta=0.001)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    h = pm.AR('h', rho=beta, sigma=pm.math.sqrt(Omega_h), shape=len(log_return_diff), constant=True)
    nu = pm.Normal('nu', mu=0, sigma=10)
    gamma = pm.Normal('gamma', mu=0, sigma=10)
    y = pm.Normal('y', mu=nu + gamma * log_positive_diff, sigma=pm.math.sqrt(pm.math.exp(h)), shape=len(log_return_diff), observed=log_return_diff)

    temp = pm.model_to_graphviz(model_T_mu)
    temp.render("20200705")

with model_T_mu:
    trace_mu = pm.sample(draws=2000, tune=1000)

sum_mu = pm.summary(trace_mu, credible_interval=0.95)
print(sum_mu)

with pm.Model() as model_T_vol:
    Omega_h = pm.InverseGamma('Omega_h', alpha=0.001, beta=0.001)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    h = pm.AR('h', rho=beta, sigma=pm.math.sqrt(Omega_h), shape=len(log_return_diff), constant=True)
    nu = pm.Normal('nu', mu=0, sigma=10)
    gamma = pm.Normal('gamma', mu=0, sigma=10)
    y = pm.StudentT('y', nu=5, mu=nu, sigma=pm.math.sqrt(pm.math.exp(h)) + gamma * log_positive_diff, shape=len(log_return_diff), observed=log_return_diff)

with model_T_vol:
    trace_vol = pm.sample(draws=2000, tune=1000)

sum_vol = pm.summary(trace_vol, credible_interval=0.95)
print(sum_vol)

plt.hist(trace_vol['gamma'], bins=80, density=True)
plt.xlabel('$\\gamma$')
plt.ylabel('density')
plt.savefig('20200707_1.pdf')
