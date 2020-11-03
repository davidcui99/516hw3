import pandas as pd
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt
from sklearn.utils import resample
from cvxopt import matrix
from cvxopt import solvers


rp = open('m_ret_10stocks.txt', 'r')
sp500 = open('m_sp500ret_3mtcm.txt', 'r')

# 2.7 a)

motor = open('w_logret_3automanu.txt', 'r')
lines = motor.readlines()[1:]

toyota = []
ford = []
gm = []

for x in lines:
    if x.split():
        toyota.append(float(x.split()[1]))
        ford.append(float(x.split()[2]))
        gm.append(float(x.split()[3]))

sample = pd.DataFrame([toyota, ford, gm])
cov = np.cov(sample)

# the sample covariance matrix is [2.59227101e-04 7.59885248e-05 6.22028379e-05]
#                                 [7.59885248e-05 3.78676652e-04 2.37339280e-04]
#                                 [6.22028379e-05 2.37339280e-04 3.95813429e-04]

# 2.7 b)

# Since we are assuming i.i.d, the true Pearson coefficients are all zeros for all pairs.

# Toyota and Ford: [0.2396944671924744, 0.245223381155803]
# Ford and GM: [0.6097455896778652, 0.6152745036411937]
# GM and Toyota: [0.19242884007433672, 0.19795775403766533]

pear_tf = stats.pearsonr(toyota, ford)[0]
pear_fg = stats.pearsonr(ford, gm)[0]
pear_tg = stats.pearsonr(toyota, gm)[0]

pear_tf_high = pear_tf + 1.96 * (1 / len(toyota))
pear_tf_low = pear_tf - 1.96 * (1 / len(toyota))

pear_fg_high = pear_fg + 1.96 * (1 / len(toyota))
pear_fg_low = pear_fg - 1.96 * (1 / len(toyota))

pear_tg_high = pear_tg + 1.96 * (1 / len(toyota))
pear_tg_low = pear_tg - 1.96 * (1 / len(toyota))

# 2.7 c)
# We are assuming each pair is i.i.d bivariate normal.
# fig, ax = plt.subplots()
# ax.plot(sample.T.index, sample.T[0], color='blue')
# ax2 = ax.twinx()
# ax2.plot(sample.T.index, sample.T[1], color='red')
# # plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(sample.T.index, sample.T[1], color='blue')
# ax2 = ax.twinx()
# ax2.plot(sample.T.index, sample.T[2], color='red')
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(sample.T.index, sample.T[0], color='blue')
# ax2 = ax.twinx()
# ax2.plot(sample.T.index, sample.T[2], color='red')
# plt.show()

# It appears that for all three graphs, the points are clustering around zero with no
# clear patterns. So we may conclude that each pair is i.i.d bivariate normal.


# 3.4

six_stocks = open('d_logret_6stocks.txt', 'r')
lines = six_stocks.readlines()[1:]

Pfizer = []
Intel = []
Citigroup = []
AmerExp = []
Exxon = []
GenMotor = []
returns = []
return1=[]
zeros = matrix([0, 0, 0, 0, 0, 0], tc='d')
dataf = [Pfizer, Intel, Citigroup, AmerExp, Exxon, GenMotor]


for x in lines:
    if x.split():
        Pfizer.append(float(x.split()[1]))
        Intel.append(float(x.split()[2]))
        Citigroup.append(float(x.split()[3]))
        AmerExp.append(float(x.split()[4]))
        Exxon.append(float(x.split()[5]))
        GenMotor.append(float(x.split()[6]))

for i in range(len(dataf)):
    dataf[i] = resample(dataf[i], replace=True, n_samples=len(Pfizer), random_state=1)
    for h in range(len(dataf[i])):
        dataf[i][h] = dataf[i][h] + 1
    returns.append([np.prod(dataf[i]),1])
    return1.append(np.prod(dataf[i]))
returns=matrix(returns,tc='d')
return1=matrix(return1,tc='d')
stocks_boot = pd.DataFrame(dataf)
stocks_cov = matrix(np.cov(stocks_boot), tc='d')
x=[-5,0,5,10,15,20]

std=[]
log_return=[]
for s in x:
    target_returns = matrix([s, 1], tc='d')
    sol = solvers.qp(stocks_cov, zeros, None, None, returns, target_returns)
    std1=sol['x'].T*stocks_cov*sol['x']
    return2=sol['x'].T*return1
    std.append(std1[0])
    log_return.append(return2[0])

# plt.xlabel('Volatility')
# plt.ylabel('Return')
# plt.plot(std,log_return)
# plt.show() The efficient frontier is included



