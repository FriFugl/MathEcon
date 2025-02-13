from turtledemo.penrose import start

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#from _vasicek_model import _short_rate_to_zcb_prices_vasicek
#from _vasicek_model import _short_rate_to_swap_rate_vasicek
from _calculate_swap_and_swaption_details import _compute_swap_rate_and_accrual_factor
from _calculate_swap_and_swaption_details import _calculate_swaption_payoffs
from _vasicek_model import VasicekModel

import statsmodels.formula.api as sm
from _LSM import _LSM_method

r_0 = 0.03
a = 0.5
b = 0.04
sigma = 0.04

T = 10
M = 10

N = 10

exercise_dates = [i for i in range(M)]

VasicekModel = VasicekModel(a=a, b=b, sigma=sigma)
short_rates = VasicekModel.simulate(r_0=r_0, T=T, M=M, N=N,method='exact')
swap_rates, accrual_factors = VasicekModel.swap_rate(short_rate=short_rates, expiry=T, exercise_dates=exercise_dates)

print(short_rates)
print(swap_rates)
print(accrual_factors)

def g():
    exercise_dates = [i/120 for i in range(M)]

    short_rates = _simulate_vasicek_short_rate(r_0=r_0, a=a, b=b, sigma=sigma, T=T,M=M, method='exact', N=N)
    swap_rates, accrual_factors = _short_rate_to_swap_rate_vasicek(short_rate=short_rates,
                                            expiry=T,
                                            exercise_dates=exercise_dates,
                                            a=a,
                                            b=b,
                                            sigma=sigma)


    payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates, accrual_factors=accrual_factors, strike=0.04)



    LSM = _LSM_method(short_rates=short_rates,swap_rates=swap_rates,accrual_factors=accrual_factors, strike=0.04, N=N)

    print(short_rates)
    print(swap_rates)
    print(accrual_factors)

    print(payoffs)
    print(LSM* 10000)

    def f():
        for i, col in enumerate(reversed(payoffs.columns)):
            discount = np.exp(-short_rates[col].iloc[0]) #NEEDS TO BE FIXED FOR TIME INTERVAL FOR DISCOUNTING

            if i == 0:
                cashflows = payoffs[col]
                continue

            itm_paths = payoffs.index[payoffs[col] > 0].tolist()
            if itm_paths == []:
                cashflows = cashflows * discount
                continue

            itm_swap_rates = swap_rates.loc[itm_paths, col]
            itm_cashflows = cashflows[itm_paths].to_numpy()

            X = np.column_stack((np.ones(len(itm_swap_rates)), itm_swap_rates, itm_swap_rates ** 2))
            beta, residuals, rank, s = np.linalg.lstsq(X, itm_cashflows, rcond=None)

            continuation_values = beta[0] + beta[1]*itm_swap_rates + beta[2]*itm_swap_rates**2
            exercised_paths = payoffs[col] > continuation_values.reindex(payoffs[col].index) * discount
            cashflows.loc[exercised_paths] = payoffs[col].loc[exercised_paths]