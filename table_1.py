from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from _LSM import LSM_method_v2

import numpy as np

def format_for_latex(data):
    latex_rows = []
    for row in data:
        x1, x2, x3, y1, err1, y2, err2, y3, err3 = row
        latex_row = (
            f"{x1:.2f} & {x2:.2f} & {x3} & "
            f"{y1:.5f} ({err1:.4f}) & {y2:.5f} ({err2:.4f}) & {y3:.5f} ({err3:.4f}) \\\\"
        )
        print(latex_row)
        latex_rows.append(latex_row)
    return "\n".join(latex_rows)

def save_all_to_latex(data, filename="output.tex"):
    with open(filename, "w") as f:
        formatted_data = format_for_latex(data)
        f.write(formatted_data + "\n")

a = 1
b = 0.04

N_calibration = 10000
N_estimation = 1000

alpha = 0.5
strike = 0.04

#(r_0, sigma, T)
combinations = [(0.02, 0.02, 5),
                (0.02, 0.02, 10),
                (0.02, 0.02, 15),
                (0.02, 0.04, 5),
                (0.02, 0.04, 10),
                (0.02, 0.04, 15),
                (0.02, 0.08, 5),
                (0.02, 0.08, 10),
                (0.02, 0.08, 15),
                (0.04, 0.02, 5),
                (0.04, 0.02, 10),
                (0.04, 0.02, 15),
                (0.04, 0.04, 5),
                (0.04, 0.04, 10),
                (0.04, 0.04, 15),
                (0.04, 0.08, 5),
                (0.04, 0.08, 10),
                (0.04, 0.08, 15),
                (0.06, 0.02, 5),
                (0.06, 0.02, 10),
                (0.06, 0.02, 15),
                (0.06, 0.04, 5),
                (0.06, 0.04, 10),
                (0.06, 0.04, 15),
                (0.06, 0.08, 5),
                (0.06, 0.08, 10),
                (0.06, 0.08, 15)
            ]

IS_table_data = []
OOS_table_data = []

for combo in combinations:
    r_0 = combo[0]
    sigma = combo[1]
    T = combo[2]
    M = T*4

    exercise_dates = [i*(T/M) for i in range(4, M+1) if i*(T/M) < T-alpha]

    VasicekModelInstance = VasicekModel(a=a, b=b, sigma=sigma)
    LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

    LSM_IS_results = []
    LSM_OOS_results = []

    M1_IS_results = []
    M1_OOS_results = []

    M2_IS_results = []
    M2_OOS_results = []

    for i in range(100):
        short_rates_calibration = VasicekModelInstance.simulate(r_0=r_0, T=T, M=M, N=N_calibration,method='exact')
        short_rates_estimation = VasicekModelInstance.simulate(r_0=r_0, T=T, M=M, N=N_estimation,method='exact')

        swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(short_rate=short_rates_calibration,
                                                             entry_dates=exercise_dates,
                                                             expiry=T,
                                                             alpha=alpha)

        swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(short_rate=short_rates_estimation,
                                                             entry_dates=exercise_dates,
                                                             expiry=T,
                                                             alpha=alpha)

        calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration, accrual_factors=accrual_factors_calibration, strike=strike)
        estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation, accrual_factors=accrual_factors_estimation, strike=strike)

        discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)
        discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

        LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                       underlying_asset_paths=swap_rates_calibration.copy(),
                                                       payoffs=calibration_payoffs.copy(),
                                                       discount_factors=discount_factors_calibration.copy())

        M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                                 underlying_asset_paths=swap_rates_calibration.copy(),
                                                 accrual_factors=accrual_factors_calibration.copy(),
                                                 payoffs=calibration_payoffs.copy(),
                                                 discount_factors=discount_factors_calibration.copy())

        M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                                 underlying_asset_paths=swap_rates_calibration.copy(),
                                                 accrual_factors=accrual_factors_calibration.copy(),
                                                 payoffs=calibration_payoffs.copy(),
                                                 discount_factors=discount_factors_calibration.copy(),
                                                 a=a)

        LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                            payoffs=estimation_payoffs.copy(),
                                            discount_factors=discount_factors_estimation.copy(),
                                            betas=LSM_betas.copy())

        M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                                         payoffs=estimation_payoffs.copy(),
                                                         discount_factors=discount_factors_estimation.copy(),
                                                         betas=M1_betas)

        M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                         payoffs=estimation_payoffs.copy(),
                                         discount_factors=discount_factors_estimation.copy(),
                                         betas=M2_betas)

        LSM_IS_results.append(LSM_IS)
        LSM_OOS_results.append(LSM_OOS)

        M1_IS_results.append(M1_IS)
        M1_OOS_results.append(M1_OOS)

        M2_IS_results.append(M2_IS)
        M2_OOS_results.append(M2_OOS)

    LSM_IS_mean = np.mean(LSM_IS_results)
    LSM_IS_sd = np.std(LSM_IS_results)

    M1_IS_mean = np.mean(M1_IS_results)
    M1_IS_sd = np.std(M1_IS_results)

    M2_IS_mean = np.mean(M2_IS_results)
    M2_IS_sd = np.std(M2_IS_results)

    LSM_OOS_mean = np.mean(LSM_OOS_results)
    LSM_OOS_sd = np.std(LSM_OOS_results)

    M1_OOS_mean = np.mean(M1_OOS_results)
    M1_OOS_sd = np.std(M1_OOS_results)

    M2_OOS_mean = np.mean(M2_OOS_results)
    M2_OOS_sd = np.std(M2_OOS_results)

    '''
    print(f"Params: r_0 = {r_0}, a={a}, b={b}, sigma={sigma}, T={T}, M={M}, strike={strike}, {N_calibration}"
          f" paths for calibration and {N_estimation} for estimation.")

    print(f"In sample mean, LSM: {LSM_IS_mean}, sd: {LSM_IS_sd}")
    print(f"In sample mean, Delta LSM (M1): {M1_IS_mean}, sd: {M1_IS_sd}")
    print(f"In sample mean, Delta LSM (M2): {M2_IS_mean}, sd: {M2_IS_sd}")

    print(f"Out of sample mean, classic LSM: {LSM_OOS_mean}, sd: {LSM_OOS_sd}")
    print(f"Out of sample mean, Delta LSM (M1): {M1_OOS_mean}, sd: {M1_OOS_sd}")
    print(f"Out of sample mean, Delta LSM (M2): {M2_OOS_mean}, sd: {M2_OOS_sd}")
    '''

    format_for_latex([(r_0, sigma, T,
                          round(LSM_IS_mean, 6),
                          round(LSM_IS_sd, 6),
                          round(M1_IS_mean, 6),
                          round(M1_IS_sd, 6),
                          round(M2_IS_mean, 6),
                          round(M2_IS_sd, 6))])

    format_for_latex([(r_0, sigma, T,
                          round(LSM_OOS_mean, 6),
                          round(LSM_OOS_sd, 6),
                          round(M1_OOS_mean, 6),
                          round(M1_OOS_sd, 6),
                          round(M2_OOS_mean, 6),
                          round(M2_OOS_sd, 6))])

save_all_to_latex(IS_table_data, filename='table_1_IS_data.tex')
save_all_to_latex(OOS_table_data, filename='table_1_OOS_data.tex')