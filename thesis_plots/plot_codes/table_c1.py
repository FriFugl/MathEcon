import pandas as pd
import numpy as np

from _helpers import _calculate_option_payoffs
from _helpers import _short_rate_to_discount_factors

from _LSM import LSM_method_v2

from _stock_path_models import GeometricBrownianMotion

def format_for_latex(data):
    latex_rows = []
    for row in data:
        x1, x2, x3, y1, err1, y2, err2, y3, err3, y4, err4 = row
        latex_row = (
            f"{x1} & {x2} & {x3} & "
            f"{y1:.3f} ({err1:.3f}) & {y2:.3f} ({err2:.3f}) & {y3:.3f} ({err3:.3f}) & {y4:.3f} ({err4:.3f}) \\\\"
        )
        print(latex_row)
        latex_rows.append(latex_row)
    return "\n".join(latex_rows)

def save_all_to_latex(data, filename="output.tex"):
    with open(filename, "w") as f:
        formatted_data = format_for_latex(data)
        f.write(formatted_data + "\n")

r = 0.06  # Risk-free rate
strike = 40  # Option strike
N_calibration = int(65536 / 2)  # Number of simulated In-sample stock paths
N_estimation = 1000 #65536 / 2  # Number of simulated In-sample stock paths

combinations = [
    (36, 0.2, 1),
    (36, 0.2, 2),
    (36, 0.4, 1),
    (36, 0.4, 2),
    (38, 0.2, 1),
    (38, 0.2, 2),
    (38, 0.4, 1),
    (38, 0.4, 2),
    (40, 0.2, 1),
    (40, 0.2, 2),
    (40, 0.4, 1),
    (40, 0.4, 2),
    (42, 0.2, 1),
    (42, 0.2, 2),
    (42, 0.4, 1),
    (42, 0.4, 2),
    (44, 0.2, 1),
    (44, 0.2, 2),
    (44, 0.4, 1),
    (44, 0.4, 2),
]


for combo in combinations:

    LSM_IS_results = []
    LSM_OOS_results = []

    delta_LSM_IS_results = []
    delta_LSM_OOS_results = []

    for i in range(100):
        s_0, sigma, T = combo[0], combo[1], combo[2]

        M = T * 50  # Total number of exercise points (50 per year)

        exercise_dates = [i * (T / M) for i in range(M + 1)]  # Exercise dates

        GBM = GeometricBrownianMotion(r=r, sigma=sigma)
        stock_paths_IS = GBM.simulate(s_0=s_0, T=T, M=M, N=N_calibration)
        stock_paths_OOS = GBM.simulate(s_0=s_0, T=T, M=M, N=N_estimation)

        short_rate_IS = pd.DataFrame(r, index=stock_paths_IS.index, columns=stock_paths_IS.columns)
        discount_factors_IS = _short_rate_to_discount_factors(short_rates=short_rate_IS)

        short_rate_OOS = pd.DataFrame(r, index=stock_paths_OOS.index, columns=stock_paths_OOS.columns)
        discount_factors_OOS = _short_rate_to_discount_factors(short_rates=short_rate_OOS)

        LSM_model = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)
        calibration_payoffs = _calculate_option_payoffs(stock_paths=stock_paths_IS, strike=strike, call=False)
        estimation_payoffs = _calculate_option_payoffs(stock_paths=stock_paths_OOS, strike=strike, call=False)

        LSM_IS, LSM_betas = LSM_model.calibration(method='classic',
                                                underlying_asset_paths=stock_paths_IS.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_IS.copy())

        delta_LSM_IS, delta_LSM_betas = LSM_model.calibration(method='stock_delta',
                                                           underlying_asset_paths=stock_paths_IS.copy(),
                                                           payoffs=calibration_payoffs.copy(),
                                                           discount_factors=discount_factors_IS.copy())

        LSM_OOS = LSM_model.estimation(underlying_asset_paths=stock_paths_OOS.copy(),
                                       payoffs=estimation_payoffs.copy(),
                                       discount_factors=discount_factors_OOS.copy(),
                                       betas=LSM_betas.copy())

        delta_LSM_OOS = LSM_model.estimation(underlying_asset_paths=stock_paths_OOS.copy(),
                                       payoffs=estimation_payoffs.copy(),
                                       discount_factors=discount_factors_OOS.copy(),
                                       betas=delta_LSM_betas.copy())

        LSM_IS_results.append(LSM_IS)
        LSM_OOS_results.append(LSM_OOS)

        delta_LSM_IS_results.append(delta_LSM_IS)
        delta_LSM_OOS_results.append(delta_LSM_OOS)

    LSM_IS_mean = np.mean(LSM_IS_results)
    LSM_IS_sd = np.std(LSM_IS_results)

    LSM_OOS_mean = np.mean(LSM_OOS_results)
    LSM_OOS_sd = np.std(LSM_OOS_results)

    delta_LSM_IS_mean = np.mean(delta_LSM_IS_results)
    delta_LSM_IS_sd = np.std(delta_LSM_IS_results)

    delta_LSM_OOS_mean = np.mean(delta_LSM_OOS_results)
    delta_LSM_OOS_sd = np.std(delta_LSM_OOS_results)

    format_for_latex([(s_0, sigma, T,
                       round(LSM_IS_mean, 3),
                       round(LSM_IS_sd, 3),
                       round(delta_LSM_IS_mean, 3),
                       round(delta_LSM_IS_sd, 3),
                       round(LSM_OOS_mean, 3),
                       round(LSM_OOS_sd, 3),
                       round(delta_LSM_OOS_mean, 3),
                       round(delta_LSM_OOS_sd, 3))])