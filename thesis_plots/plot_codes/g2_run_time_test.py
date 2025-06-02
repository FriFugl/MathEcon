from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
import time

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_eta
from thesis_plots.plot_codes._parameter_config import g2_rho

from thesis_plots.plot_codes._parameter_config import colors

def simul_run_time():
    N_calibration = 10000
    N_estimation = 1000

    backwards_pass_simulation_time_T = []
    backwards_pass_simulation_time_T_sd = []

    forward_pass_simulation_time_T = []
    forward_pass_simulation_time_T_sd = []

    LSM_calibration_times_T = []
    M1_calibration_times_T = []

    LSM_calibration_times_T_sd = []
    M1_calibration_times_T_sd = []

    LSM_estimation_times_T = []
    M1_estimation_times_T = []

    LSM_estimation_times_T_sd = []
    M1_estimation_times_T_sd = []

    backwards_pass_simulation_time_k = []
    backwards_pass_simulation_time_k_sd = []

    forward_pass_simulation_time_k = []
    forward_pass_simulation_time_k_sd = []

    LSM_calibration_times_k = []
    M1_calibration_times_k = []

    LSM_calibration_times_k_sd = []
    M1_calibration_times_k_sd = []

    LSM_estimation_times_k = []
    M1_estimation_times_k = []

    LSM_estimation_times_k_sd = []
    M1_estimation_times_k_sd = []

    backwards_pass_simulation_time_N_calibration = []
    backwards_pass_simulation_time_N_calibration_sd = []

    forward_pass_simulation_time_N_calibration = []
    forward_pass_simulation_time_N_calibration_sd = []

    LSM_calibration_times_N_calibration = []
    M1_calibration_times_N_calibration = []

    LSM_calibration_times_N_calibration_sd = []
    M1_calibration_times_N_calibration_sd = []

    LSM_estimation_times_N_calibration = []
    M1_estimation_times_N_calibration = []

    LSM_estimation_times_N_calibration_sd = []
    M1_estimation_times_N_calibration_sd = []

    backwards_pass_simulation_time_N_estimation = []
    backwards_pass_simulation_time_N_estimation_sd = []

    forward_pass_simulation_time_N_estimation = []
    forward_pass_simulation_time_N_estimation_sd = []

    LSM_calibration_times_N_estimation = []
    M1_calibration_times_N_estimation = []

    LSM_calibration_times_N_estimation_sd = []
    M1_calibration_times_N_estimation_sd = []

    LSM_estimation_times_N_estimation = []
    M1_estimation_times_N_estimation = []

    LSM_estimation_times_N_estimation_sd = []
    M1_estimation_times_N_estimation_sd = []

    for T in [5, 10, 15]:
        T = T
        M = T * 1

        backwards_pass_simulation_times = []
        forward_pass_simulation_times = []

        LSM_calibration = []
        M1_calibration = []

        LSM_estimation = []
        M1_estimation = []

        instant_forward_rates = dict(zip(maturities, market_forward_rates))
        GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                              instant_forward_rates=instant_forward_rates)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)
        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(
                T=T,
                M=M,
                N=N_calibration,
                method='euler')

            swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                                  y_paths=y_calibration,
                                                                                                  varphi=varphi_calibration,
                                                                                                  entry_dates=exercise_dates,
                                                                                                  expiry=T,
                                                                                                  alpha=alpha)

            calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                              accrual_factors=accrual_factors_calibration,
                                                              strike=strike)
            discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)

            backwards_pass_simulation_times.append(time.time() - backwards_pass_simulation_start)

            calibration_start = time.time()

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())
            LSM_calibration.append(time.time() - calibration_start)

            calibration_start = time.time()
            M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              accrual_factors=accrual_factors_calibration.copy())
            M1_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T,
                                                                                                                   M=M,
                                                                                                                   N=N_estimation,
                                                                                                                   method='euler')

            swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                                y_paths=y_estimation,
                                                                                                varphi=varphi_estimation,
                                                                                                entry_dates=exercise_dates,
                                                                                                expiry=T,
                                                                                                alpha=alpha)

            estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                             accrual_factors=accrual_factors_estimation, strike=strike)

            discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

            forward_pass_simulation_times.append(time.time() - forward_pass_simulation_start)

            LSM_estimation_start = time.time()

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            LSM_estimation.append(time.time() - LSM_estimation_start)

            M1_estimation_start = time.time()

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())
            M1_estimation.append(time.time() - M1_estimation_start)

        backwards_pass_simulation_time_T.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_T_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_T.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_T_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_T.append(np.mean(LSM_calibration))
        M1_calibration_times_T.append(np.mean(M1_calibration))

        LSM_calibration_times_T_sd.append(np.std(LSM_calibration))
        M1_calibration_times_T_sd.append(np.std(M1_calibration))

        LSM_estimation_times_T.append(np.mean(LSM_estimation))
        M1_estimation_times_T.append(np.mean(M1_estimation))

        LSM_estimation_times_T_sd.append(np.std(LSM_estimation))
        M1_estimation_times_T_sd.append(np.std(M1_estimation))

    print(f"Varying T:")
    print(f"backwards_pass_simulation_time_T = {backwards_pass_simulation_time_T}")
    print(f"forward_pass_simulation_time_T = {forward_pass_simulation_time_T}")

    print(f"LSM_calibration_times_T = {LSM_calibration_times_T}")
    print(f"M1_calibration_times_T = {M1_calibration_times_T}")
    print(f"LSM_estimation_times_T = {LSM_estimation_times_T}")
    print(f"M1_estimation_times_T = {M1_estimation_times_T}")

    for k in [1, 2, 4, 12]:
        T = 10
        M = T * k

        backwards_pass_simulation_times = []
        forward_pass_simulation_times = []

        LSM_calibration = []
        M1_calibration = []

        LSM_estimation = []
        M1_estimation = []

        instant_forward_rates = dict(zip(maturities, market_forward_rates))
        GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                              instant_forward_rates=instant_forward_rates)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)
        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(
                T=T,
                M=M,
                N=N_calibration,
                method='euler')

            swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                                  y_paths=y_calibration,
                                                                                                  varphi=varphi_calibration,
                                                                                                  entry_dates=exercise_dates,
                                                                                                  expiry=T,
                                                                                                  alpha=alpha)

            calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                              accrual_factors=accrual_factors_calibration,
                                                              strike=strike)

            discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)

            backwards_pass_simulation_times.append(time.time() - backwards_pass_simulation_start)

            calibration_start = time.time()

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())
            LSM_calibration.append(time.time() - calibration_start)

            calibration_start = time.time()
            M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              accrual_factors=accrual_factors_calibration.copy())
            M1_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T,
                                                                                                                   M=M,
                                                                                                                   N=N_estimation,
                                                                                                                   method='euler')

            swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                                y_paths=y_estimation,
                                                                                                varphi=varphi_estimation,
                                                                                                entry_dates=exercise_dates,
                                                                                                expiry=T,
                                                                                                alpha=alpha)

            estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                             accrual_factors=accrual_factors_estimation,
                                                             strike=strike)

            discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

            forward_pass_simulation_times.append(time.time() - forward_pass_simulation_start)

            LSM_estimation_start = time.time()

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            LSM_estimation.append(time.time() - LSM_estimation_start)

            M1_estimation_start = time.time()

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())
            M1_estimation.append(time.time() - M1_estimation_start)

        backwards_pass_simulation_time_k.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_k_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_k.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_k_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_k.append(np.mean(LSM_calibration))
        M1_calibration_times_k.append(np.mean(M1_calibration))

        LSM_calibration_times_k_sd.append(np.std(LSM_calibration))
        M1_calibration_times_k_sd.append(np.std(M1_calibration))

        LSM_estimation_times_k.append(np.mean(LSM_estimation))
        M1_estimation_times_k.append(np.mean(M1_estimation))

        LSM_estimation_times_k_sd.append(np.std(LSM_estimation))
        M1_estimation_times_k_sd.append(np.std(M1_estimation))

    print(f"Varying k:")
    print(f"backwards_pass_simulation_time_k = {backwards_pass_simulation_time_k}")
    print(f"forward_pass_simulation_time_k = {forward_pass_simulation_time_k}")

    print(f"LSM_calibration_times_k = {LSM_calibration_times_k}")
    print(f"M1_calibration_times_k = {M1_calibration_times_k}")
    print(f"LSM_estimation_times_k = {LSM_estimation_times_k}")
    print(f"M1_estimation_times_k = {M1_estimation_times_k}")

    for N in [100, 1000, 10000, 100000]:
        T = T
        M = T * 1

        backwards_pass_simulation_times = []
        forward_pass_simulation_times = []

        LSM_calibration = []
        M1_calibration = []
        M2_calibration = []

        LSM_estimation = []
        M1_estimation = []
        M2_estimation = []

        instant_forward_rates = dict(zip(maturities, market_forward_rates))
        GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                              instant_forward_rates=instant_forward_rates)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(
                T=T,
                M=M,
                N=N,
                method='euler')

            swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                                  y_paths=y_calibration,
                                                                                                  varphi=varphi_calibration,
                                                                                                  entry_dates=exercise_dates,
                                                                                                  expiry=T,
                                                                                                  alpha=alpha)

            calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                              accrual_factors=accrual_factors_calibration,
                                                              strike=strike)

            discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)

            backwards_pass_simulation_times.append(time.time() - backwards_pass_simulation_start)

            calibration_start = time.time()

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())
            LSM_calibration.append(time.time() - calibration_start)

            calibration_start = time.time()
            M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              accrual_factors=accrual_factors_calibration.copy())
            M1_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T,
                                                                                                                   M=M,
                                                                                                                   N=N_estimation,
                                                                                                                   method='euler')

            swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                                y_paths=y_estimation,
                                                                                                varphi=varphi_estimation,
                                                                                                entry_dates=exercise_dates,
                                                                                                expiry=T,
                                                                                                alpha=alpha)

            estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                             accrual_factors=accrual_factors_estimation,
                                                             strike=strike)

            discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

            forward_pass_simulation_times.append(time.time() - forward_pass_simulation_start)

            LSM_estimation_start = time.time()

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            LSM_estimation.append(time.time() - LSM_estimation_start)

            M1_estimation_start = time.time()

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())
            M1_estimation.append(time.time() - M1_estimation_start)

        backwards_pass_simulation_time_N_calibration.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_N_calibration_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_N_calibration.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_N_calibration_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_N_calibration.append(np.mean(LSM_calibration))
        M1_calibration_times_N_calibration.append(np.mean(M1_calibration))

        LSM_calibration_times_N_calibration_sd.append(np.std(LSM_calibration))
        M1_calibration_times_N_calibration_sd.append(np.std(M1_calibration))

        LSM_estimation_times_N_calibration.append(np.mean(LSM_estimation))
        M1_estimation_times_N_calibration.append(np.mean(M1_estimation))

        LSM_estimation_times_N_calibration_sd.append(np.std(LSM_estimation))
        M1_estimation_times_N_calibration_sd.append(np.std(M1_estimation))

    print(f"Varying calibration N:")
    print(f"backwards_pass_simulation_time_N_calibration = {backwards_pass_simulation_time_N_calibration}")
    print(f"forward_pass_simulation_time_N_calibration = {forward_pass_simulation_time_N_calibration}")

    print(f"LSM_calibration_times_N_calibration = {LSM_calibration_times_N_calibration}")
    print(f"M1_calibration_times_N_calibration = {M1_calibration_times_N_calibration}")
    print(f"LSM_estimation_times_N_calibration = {LSM_estimation_times_N_calibration}")
    print(f"M1_estimation_times_N_calibration = {M1_estimation_times_N_calibration}")

    for N in [100, 1000, 10000, 100000]:
        T = 10
        M = T * 1

        backwards_pass_simulation_times = []
        forward_pass_simulation_times = []

        LSM_calibration = []
        M1_calibration = []

        LSM_estimation = []
        M1_estimation = []

        instant_forward_rates = dict(zip(maturities, market_forward_rates))
        GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                              instant_forward_rates=instant_forward_rates)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=T, M=M, N=N_calibration, method='euler')

            swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                                  y_paths=y_calibration,
                                                                                                  varphi=varphi_calibration,
                                                                                                  entry_dates=exercise_dates,
                                                                                                  expiry=T,
                                                                                                  alpha=alpha)

            calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                              accrual_factors=accrual_factors_calibration,
                                                              strike=strike)

            discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)

            backwards_pass_simulation_times.append(time.time() - backwards_pass_simulation_start)

            calibration_start = time.time()

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())
            LSM_calibration.append(time.time() - calibration_start)

            calibration_start = time.time()
            M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              accrual_factors=accrual_factors_calibration.copy())
            M1_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T, M=M, N=N, method='euler')

            swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                                y_paths=y_estimation,
                                                                                                varphi=varphi_estimation,
                                                                                                entry_dates=exercise_dates,
                                                                                                expiry=T,
                                                                                                alpha=alpha)

            estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                             accrual_factors=accrual_factors_estimation,
                                                             strike=strike)

            discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

            forward_pass_simulation_times.append(time.time() - forward_pass_simulation_start)

            LSM_estimation_start = time.time()

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            LSM_estimation.append(time.time() - LSM_estimation_start)

            M1_estimation_start = time.time()

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())
            M1_estimation.append(time.time() - M1_estimation_start)

        backwards_pass_simulation_time_N_estimation.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_N_estimation_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_N_estimation.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_N_estimation_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_N_estimation.append(np.mean(LSM_calibration))
        M1_calibration_times_N_estimation.append(np.mean(M1_calibration))

        LSM_calibration_times_N_estimation_sd.append(np.std(LSM_calibration))
        M1_calibration_times_N_estimation_sd.append(np.std(M1_calibration))

        LSM_estimation_times_N_estimation.append(np.mean(LSM_estimation))
        M1_estimation_times_N_estimation.append(np.mean(M1_estimation))

        LSM_estimation_times_N_estimation_sd.append(np.std(LSM_estimation))
        M1_estimation_times_N_estimation_sd.append(np.std(M1_estimation))

    print(f"Varying estimation N:")
    print(f"backwards_pass_simulation_time_N_estimation = {backwards_pass_simulation_time_N_estimation}")
    print(f"forward_pass_simulation_time_N_estimation = {forward_pass_simulation_time_N_estimation}")

    print(f"LSM_calibration_times_N_estimation = {LSM_calibration_times_N_estimation}")
    print(f"M1_calibration_times_N_estimation = {M1_calibration_times_N_estimation}")
    print(f"LSM_estimation_times_N_estimation = {LSM_estimation_times_N_estimation}")
    print(f"M1_estimation_times_N_estimation = {M1_estimation_times_N_estimation}")

backwards_pass_simulation_time_T = [np.float64(0.029283208847045897),
                                    np.float64(0.08099297523498535),
                                    np.float64(0.16184576749801635)]

forward_pass_simulation_time_T = [np.float64(0.013776953220367432),
                                  np.float64(0.04532495498657227),
                                  np.float64(0.09504778861999512)]

LSM_calibration_times_T = [np.float64(0.04327594995498657),
                           np.float64(0.18177083253860474),
                           np.float64(0.32035250186920167)]

M1_calibration_times_T = [np.float64(0.05641968250274658),
                          np.float64(0.19116333484649659),
                          np.float64(0.3896254348754883)]

LSM_estimation_times_T = [np.float64(0.006351685523986817),
                          np.float64(0.0160345721244812),
                          np.float64(0.025320026874542236)]

M1_estimation_times_T = [np.float64(0.006196780204772949),
                         np.float64(0.015752747058868408),
                         np.float64(0.0247953462600708)]

backwards_pass_simulation_time_k = [np.float64(0.08154947996139526),
                                    np.float64(0.1921987009048462),
                                    np.float64(0.3610746502876282),
                                    np.float64(1.1242478442192079)]

forward_pass_simulation_time_k = [np.float64(0.04557289123535156),
                                  np.float64(0.10219822168350219),
                                  np.float64(0.20045511722564696),
                                  np.float64(0.6173872637748719)]

LSM_calibration_times_k = [np.float64(0.18488317251205444),
                           np.float64(0.44681047439575194),
                           np.float64(0.8985964179039001),
                           np.float64(3.0434778475761415)]

M1_calibration_times_k = [np.float64(0.19333568334579468),
                          np.float64(0.5341443133354187),
                          np.float64(1.1068325161933898),
                          np.float64(3.6618267273902894)]

LSM_estimation_times_k = [np.float64(0.01601651430130005),
                          np.float64(0.03712092876434326),
                          np.float64(0.06685108423233033),
                          np.float64(0.20203089952468872)]

M1_estimation_times_k = [np.float64(0.015863573551177977),
                         np.float64(0.03555102825164795),
                         np.float64(0.06683212757110596),
                         np.float64(0.2004205846786499)]

backwards_pass_simulation_time_N_calibration = [np.float64(0.09982162714004517),
                                                np.float64(0.04762840270996094),
                                                np.float64(0.08216288328170776),
                                                np.float64(0.5289106893539429)]

forward_pass_simulation_time_N_calibration = [np.float64(0.09648611307144166),
                                              np.float64(0.046539721488952634),
                                              np.float64(0.04593893527984619),
                                              np.float64(0.0494016695022583)]

LSM_calibration_times_N_calibration = [np.float64(0.12135479927062988),
                                       np.float64(0.08561003923416138),
                                       np.float64(0.19031927824020387),
                                       np.float64(1.2478608584403992)]

M1_calibration_times_N_calibration = [np.float64(0.23178940296173095),
                                      np.float64(0.1152725625038147),
                                      np.float64(0.1943622946739197),
                                      np.float64(1.8356901597976685)]

LSM_estimation_times_N_calibration = [np.float64(0.03875712633132935),
                                      np.float64(0.01659841775894165),
                                      np.float64(0.016197826862335205),
                                      np.float64(0.017481825351715087)]

M1_estimation_times_N_calibration = [np.float64(0.03816066980361939),
                                     np.float64(0.016645703315734863),
                                     np.float64(0.01605929374694824),
                                     np.float64(0.01722888708114624)]

backwards_pass_simulation_time_N_estimation = [np.float64(0.09148058176040649),
                                               np.float64(0.09664896011352539),
                                               np.float64(0.08679981946945191),
                                               np.float64(0.09207556962966919)]

forward_pass_simulation_time_N_estimation = [np.float64(0.04576610088348389),
                                             np.float64(0.05381072044372558),
                                             np.float64(0.08811207056045532),
                                             np.float64(0.5582164716720581)]

LSM_calibration_times_N_estimation = [np.float64(0.20926363945007323),
                                      np.float64(0.219718599319458),
                                      np.float64(0.19624711513519288),
                                      np.float64(0.20551644563674926)]

M1_calibration_times_N_estimation = [np.float64(0.2142048716545105),
                                     np.float64(0.22870441436767577),
                                     np.float64(0.20951992273330688),
                                     np.float64(0.21235989332199096)]

LSM_estimation_times_N_estimation = [np.float64(0.01618985891342163),
                                     np.float64(0.01894942045211792),
                                     np.float64(0.02744638204574585),
                                     np.float64(0.14217886209487915)]

M1_estimation_times_N_estimation = [np.float64(0.015733680725097655),
                                    np.float64(0.018495147228240968),
                                    np.float64(0.02758406162261963),
                                    np.float64(0.14263400077819824)]

#Run time for a 10Y No-Call 1 Bermudan swaption with 10,000 calibration paths and 1,000 estimation paths
backwards_pass_time = backwards_pass_simulation_time_T[1]
forwards_pass_time = forward_pass_simulation_time_T[1]

calibration_time = [LSM_calibration_times_T[1], M1_calibration_times_T[1]]
estimation_time = [LSM_estimation_times_T[1], M1_estimation_times_T[1]]

bottom_calibration = np.add(backwards_pass_time, forwards_pass_time)
bottom_estimation = np.add(bottom_calibration, calibration_time)

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

r = [0.2,1]
names = ['LSM','Delta M1']
barWidth = 0.5

plt.bar(r, backwards_pass_time, color=colors['dark_blue'], edgecolor='white', width=barWidth, label='Simulation of In-sample paths')
plt.bar(r, forwards_pass_time, bottom=backwards_pass_time, color=colors['dark_petrol'], edgecolor='white', width=barWidth,
        label='Simulation of Out-of-sample paths')

plt.bar(r, calibration_time, bottom=bottom_calibration, color=colors['dark_green'], edgecolor='white', width=barWidth, label='Backwards pss')
plt.bar(r, estimation_time, bottom=bottom_estimation, color=colors['dark_grey'], edgecolor='white', width=barWidth, label='Forward pass')

plt.xticks(r, names)
ax.set_yticks([0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

plt.savefig(fr'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\g2_run_time_test.png',
          bbox_inches='tight')
plt.show()
