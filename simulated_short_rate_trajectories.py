from _helpers import _calculate_swaption_payoffs
from _helpers import _calculate_option_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from matplotlib import pyplot as plt
from _LSM import LSM_method

import numpy as np
import pandas as pd

from _stock_path_models import GeometricBrownianMotion

def swaption_test():
    r_0 = 0.03
    a = 1
    b = 0.05
    sigma = 0.04

    T = 10
    M = 120

    N = 1000

    alpha = 1

    #exercise_dates = [i for i in range(9)]
    exercise_dates = [i*(T/M) for i in range(M+1) if i*(T/M) < T-alpha]

    VasicekModel = VasicekModel(a=a, b=b, sigma=sigma)
    short_rates_calibration = VasicekModel.simulate(r_0=r_0, T=T, M=M, N=N,method='exact', seed=10)
    short_rates_estimation = VasicekModel.simulate(r_0=r_0, T=T, M=M, N=N,method='exact', seed=10)

    #print(sum(short_rates[T])/N)

    swap_rates_calbration, accrual_factors_calibration = VasicekModel.swap_rate(short_rate=short_rates_calibration,
                                                         entry_dates=exercise_dates,
                                                         expiry=T,
                                                         alpha=alpha)

    swap_rates_estimation, accrual_factors_estimation = VasicekModel.swap_rate(short_rate=short_rates_estimation,
                                                         entry_dates=exercise_dates,
                                                         expiry=T,
                                                         alpha=alpha)

    strike = 0.04
    LSM_model = LSM_method(strike=strike, exercise_dates=exercise_dates, basis_function=('hermite',5))

    calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calbration, accrual_factors=accrual_factors_calibration, strike=strike)
    estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation, accrual_factors=accrual_factors_estimation, strike=strike)

    backwards_induction, fitted_basis_functions = LSM_model.calibration(short_rate=short_rates_calibration,
                                                       underlying_asset_paths=swap_rates_calbration,
                                                       payoffs=calibration_payoffs)

    forward_pass = LSM_model.estimation(short_rate=short_rates_estimation,
                                        underlying_asset_paths=swap_rates_estimation,
                                        payoffs=estimation_payoffs,
                                        fitted_basis_functions=fitted_basis_functions)


    print(backwards_induction*10000)



    print(forward_pass*10000)

    payoffs = _calculate_swaption_payoffs(
                swap_rates=swap_rates_estimation, accrual_factors=accrual_factors_estimation, strike=strike
            )

    max_payoffs = payoffs.max(axis=1)

    print((max_payoffs.to_numpy().sum()/N)*10000)

#import time
#start = time.time()
def option_test():
    r = 0.06
    sigma = 0.4
    strike = 40

    s_0 = 36

    T = 2
    M = T*50

    N = 100000

    exercise_dates = [i*(T/M) for i in range(M+1)]
    GBM = GeometricBrownianMotion(r=r, sigma=sigma)
    stock_paths = GBM.simulate(s_0=s_0, T=T, M=M, N=N)

    short_rate = pd.DataFrame(r, index=stock_paths.index, columns=stock_paths.columns)
    discount_factors = _short_rate_to_discount_factors(short_rates=short_rate)
    """
    exercise_dates = [0,1,2,3]
    data = {0: pd.Series([1,1,1,1,1,1,1,1],
                           index=[1,2,3,4,5,6,7,8]),
            1: pd.Series([1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88],
                           index=[1,2,3,4,5,6,7,8]),
            2: pd.Series([1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22],
                           index=[1,2,3,4,5,6,7,8]),
            3: pd.Series([1.34, 1.54, 1.03, 0.92, 1.52, 0.9, 1.01, 1.34],
                           index=[1,2,3,4,5,6,7,8])
    }
    stock_paths = pd.DataFrame(data)
    """

    LSM_model = LSM_method(strike=strike, exercise_dates=exercise_dates, basis_function=('laguerre',3))

    calibration_payoffs = _calculate_option_payoffs(stock_paths=stock_paths, strike=strike, call=False)


    backwards_induction, fitted_basis_functions = LSM_model.calibration(underlying_asset_paths=stock_paths,
                                                       payoffs=calibration_payoffs,
                                                       discount_factors=discount_factors)

    forward_pass = LSM_model.estimation(underlying_asset_paths=stock_paths,
                                        payoffs=calibration_payoffs,
                                        discount_factors=discount_factors,
                                        fitted_basis_functions=fitted_basis_functions)

    #print(backwards_induction)

    end = time.time()
    print(f"LSM result: {backwards_induction, forward_pass}, run-time: {end-start}")


VasicekModel = VasicekModel(a=1, b=0.05, sigma=0.04)
simulated_short_rates = VasicekModel.simulate(r_0=0.03, T=10, M=120, N=5, method='exact', seed=10)
print(simulated_short_rates)
plt.figure(figsize=(8, 5))
for i, row in simulated_short_rates.iterrows():
    plt.plot(simulated_short_rates.columns, row, label=f'Trajectory {i+1}', alpha=0.7)
plt.xlabel('$t$', fontsize=15)
plt.ylabel('$r_{t}$', fontsize=15).set_rotation(0)
plt.title('Simulation of 5 short rates in the Vasiček model, seed = 10', fontsize=15)
plt.show()
