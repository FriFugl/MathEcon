from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
import time

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import vasicek_r_0
from thesis_plots.plot_codes._parameter_config import vasicek_a
from thesis_plots.plot_codes._parameter_config import vasicek_b
from thesis_plots.plot_codes._parameter_config import vasicek_sigma

def simul_run_time():
    N_calibration = 10000
    N_estimation= 1000

    backwards_pass_simulation_time_T = []
    backwards_pass_simulation_time_T_sd = []

    forward_pass_simulation_time_T = []
    forward_pass_simulation_time_T_sd = []

    LSM_calibration_times_T = []
    M1_calibration_times_T = []
    M2_calibration_times_T = []

    LSM_calibration_times_T_sd = []
    M1_calibration_times_T_sd = []
    M2_calibration_times_T_sd = []

    LSM_estimation_times_T = []
    M1_estimation_times_T = []
    M2_estimation_times_T = []

    LSM_estimation_times_T_sd = []
    M1_estimation_times_T_sd = []
    M2_estimation_times_T_sd = []

    backwards_pass_simulation_time_k = []
    backwards_pass_simulation_time_k_sd = []

    forward_pass_simulation_time_k = []
    forward_pass_simulation_time_k_sd = []

    LSM_calibration_times_k = []
    M1_calibration_times_k = []
    M2_calibration_times_k = []

    LSM_calibration_times_k_sd = []
    M1_calibration_times_k_sd = []
    M2_calibration_times_k_sd = []

    LSM_estimation_times_k = []
    M1_estimation_times_k = []
    M2_estimation_times_k = []

    LSM_estimation_times_k_sd = []
    M1_estimation_times_k_sd = []
    M2_estimation_times_k_sd = []

    backwards_pass_simulation_time_N_calibration = []
    backwards_pass_simulation_time_N_calibration_sd = []

    forward_pass_simulation_time_N_calibration = []
    forward_pass_simulation_time_N_calibration_sd = []

    LSM_calibration_times_N_calibration = []
    M1_calibration_times_N_calibration = []
    M2_calibration_times_N_calibration = []

    LSM_calibration_times_N_calibration_sd = []
    M1_calibration_times_N_calibration_sd = []
    M2_calibration_times_N_calibration_sd = []

    LSM_estimation_times_N_calibration = []
    M1_estimation_times_N_calibration = []
    M2_estimation_times_N_calibration = []

    LSM_estimation_times_N_calibration_sd = []
    M1_estimation_times_N_calibration_sd = []
    M2_estimation_times_N_calibration_sd = []

    backwards_pass_simulation_time_N_estimation = []
    backwards_pass_simulation_time_N_estimation_sd = []

    forward_pass_simulation_time_N_estimation = []
    forward_pass_simulation_time_N_estimation_sd = []

    LSM_calibration_times_N_estimation = []
    M1_calibration_times_N_estimation = []
    M2_calibration_times_N_estimation = []

    LSM_calibration_times_N_estimation_sd = []
    M1_calibration_times_N_estimation_sd = []
    M2_calibration_times_N_estimation_sd = []

    LSM_estimation_times_N_estimation = []
    M1_estimation_times_N_estimation = []
    M2_estimation_times_N_estimation = []

    LSM_estimation_times_N_estimation_sd = []
    M1_estimation_times_N_estimation_sd = []
    M2_estimation_times_N_estimation_sd = []

    for T in [5, 10, 15]:
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

        VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)
        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_calibration,
                                                                    method='exact')

            swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
                short_rate=short_rates_calibration,
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

            calibration_start = time.time()
            M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              a=vasicek_a)
            M2_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_estimation, method='exact')

            swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
                short_rate=short_rates_estimation,
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

            M2_estimation_start = time.time()

            M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M2_betas.copy())

            M2_estimation.append(time.time() - M2_estimation_start)

        backwards_pass_simulation_time_T.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_T_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_T.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_T_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_T.append(np.mean(LSM_calibration))
        M1_calibration_times_T.append(np.mean(M1_calibration))
        M2_calibration_times_T.append(np.mean(M2_calibration))

        LSM_calibration_times_T_sd.append(np.std(LSM_calibration))
        M1_calibration_times_T_sd.append(np.std(M1_calibration))
        M2_calibration_times_T_sd.append(np.std(M2_calibration))

        LSM_estimation_times_T.append(np.mean(LSM_estimation))
        M1_estimation_times_T.append(np.mean(M1_estimation))
        M2_estimation_times_T.append(np.mean(M2_estimation))

        LSM_estimation_times_T_sd.append(np.std(LSM_estimation))
        M1_estimation_times_T_sd.append(np.std(M1_estimation))
        M2_estimation_times_T_sd.append(np.std(M2_estimation))

    print(f"Varying T:")
    print(f"backwards_pass_simulation_time_T = {backwards_pass_simulation_time_T}")
    print(f"forward_pass_simulation_time_T = {forward_pass_simulation_time_T}")

    print(f"LSM_calibration_times_T = {LSM_calibration_times_T}")
    print(f"M1_calibration_times_T = {M1_calibration_times_T}")
    print(f"M2_calibration_times_T = {M2_calibration_times_T}")
    print(f"LSM_estimation_times_T = {LSM_estimation_times_T}")
    print(f"M1_estimation_times_T = {M1_estimation_times_T}")
    print(f"M2_estimation_times_T = {M2_estimation_times_T}")

    for k in [1, 2, 4, 12]:
        T = 10
        M = T * k

        backwards_pass_simulation_times = []
        forward_pass_simulation_times = []

        LSM_calibration = []
        M1_calibration = []
        M2_calibration = []

        LSM_estimation = []
        M1_estimation = []
        M2_estimation = []

        VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)
        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0,
                                                                    T=T,
                                                                    M=M,
                                                                    N=N_calibration,
                                                                    method='exact'
                                                                    )

            swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
                short_rate=short_rates_calibration,
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

            calibration_start = time.time()
            M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              a=vasicek_a)
            M2_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0,
                                                                   T=T,
                                                                   M=M,
                                                                   N=N_estimation,
                                                                   method='exact'
                                                                   )

            swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
                                                                                     short_rate=short_rates_estimation,
                                                                                     entry_dates=exercise_dates,
                                                                                     expiry=T,
                                                                                     alpha=alpha
                                                                                    )

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

            M2_estimation_start = time.time()

            M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M2_betas.copy())

            M2_estimation.append(time.time() - M2_estimation_start)

        backwards_pass_simulation_time_k.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_k_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_k.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_k_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_k.append(np.mean(LSM_calibration))
        M1_calibration_times_k.append(np.mean(M1_calibration))
        M2_calibration_times_k.append(np.mean(M2_calibration))

        LSM_calibration_times_k_sd.append(np.std(LSM_calibration))
        M1_calibration_times_k_sd.append(np.std(M1_calibration))
        M2_calibration_times_k_sd.append(np.std(M2_calibration))

        LSM_estimation_times_k.append(np.mean(LSM_estimation))
        M1_estimation_times_k.append(np.mean(M1_estimation))
        M2_estimation_times_k.append(np.mean(M2_estimation))

        LSM_estimation_times_k_sd.append(np.std(LSM_estimation))
        M1_estimation_times_k_sd.append(np.std(M1_estimation))
        M2_estimation_times_k_sd.append(np.std(M2_estimation))

    print(f"Varying k:")
    print(f"backwards_pass_simulation_time_k = {backwards_pass_simulation_time_k}")
    print(f"forward_pass_simulation_time_k = {forward_pass_simulation_time_k}")

    print(f"LSM_calibration_times_k = {LSM_calibration_times_k}")
    print(f"M1_calibration_times_k = {M1_calibration_times_k}")
    print(f"M2_calibration_times_k = {M2_calibration_times_k}")
    print(f"LSM_estimation_times_k = {LSM_estimation_times_k}")
    print(f"M1_estimation_times_k = {M1_estimation_times_k}")
    print(f"M2_estimation_times_k = {M2_estimation_times_k}")

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

        VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N,
                                                                    method='exact'
                                                                    )

            swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
                short_rate=short_rates_calibration,
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

            calibration_start = time.time()
            M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              a=vasicek_a)
            M2_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_estimation, method='exact')

            swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
                                                                                     short_rate=short_rates_estimation,
                                                                                     entry_dates=exercise_dates,
                                                                                     expiry=T,
                                                                                     alpha=alpha
                                                                                    )

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

            M2_estimation_start = time.time()

            M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M2_betas.copy())

            M2_estimation.append(time.time() - M2_estimation_start)

        backwards_pass_simulation_time_N_calibration.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_N_calibration_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_N_calibration.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_N_calibration_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_N_calibration.append(np.mean(LSM_calibration))
        M1_calibration_times_N_calibration.append(np.mean(M1_calibration))
        M2_calibration_times_N_calibration.append(np.mean(M2_calibration))

        LSM_calibration_times_N_calibration_sd.append(np.std(LSM_calibration))
        M1_calibration_times_N_calibration_sd.append(np.std(M1_calibration))
        M2_calibration_times_N_calibration_sd.append(np.std(M2_calibration))

        LSM_estimation_times_N_calibration.append(np.mean(LSM_estimation))
        M1_estimation_times_N_calibration.append(np.mean(M1_estimation))
        M2_estimation_times_N_calibration.append(np.mean(M2_estimation))

        LSM_estimation_times_N_calibration_sd.append(np.std(LSM_estimation))
        M1_estimation_times_N_calibration_sd.append(np.std(M1_estimation))
        M2_estimation_times_N_calibration_sd.append(np.std(M2_estimation))

    print(f"Varying calibration N:")
    print(f"backwards_pass_simulation_time_N_calibration = {backwards_pass_simulation_time_N_calibration}")
    print(f"forward_pass_simulation_time_N_calibration = {forward_pass_simulation_time_N_calibration}")

    print(f"LSM_calibration_times_N_calibration = {LSM_calibration_times_N_calibration}")
    print(f"M1_calibration_times_N_calibration = {M1_calibration_times_N_calibration}")
    print(f"M2_calibration_times_N_calibration = {M2_calibration_times_N_calibration}")
    print(f"LSM_estimation_times_N_calibration = {LSM_estimation_times_N_calibration}")
    print(f"M1_estimation_times_N_calibration = {M1_estimation_times_N_calibration}")
    print(f"M2_estimation_times_N_calibration = {M2_estimation_times_N_calibration}")

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

        VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

        exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        for i in range(100):
            backwards_pass_simulation_start = time.time()
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_calibration,
                                                                    method='exact'
                                                                    )

            swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
                short_rate=short_rates_calibration,
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

            calibration_start = time.time()
            M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              a=vasicek_a)
            M2_calibration.append(time.time() - calibration_start)

            forward_pass_simulation_start = time.time()
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N,
                                                                   method='exact')

            swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
                short_rate=short_rates_estimation,
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

            M2_estimation_start = time.time()

            M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M2_betas.copy())

            M2_estimation.append(time.time() - M2_estimation_start)

        backwards_pass_simulation_time_N_estimation.append(np.mean(backwards_pass_simulation_times))
        backwards_pass_simulation_time_N_estimation_sd.append(np.std(backwards_pass_simulation_times))

        forward_pass_simulation_time_N_estimation.append(np.mean(forward_pass_simulation_times))
        forward_pass_simulation_time_N_estimation_sd.append(np.mean(forward_pass_simulation_times))

        LSM_calibration_times_N_estimation.append(np.mean(LSM_calibration))
        M1_calibration_times_N_estimation.append(np.mean(M1_calibration))
        M2_calibration_times_N_estimation.append(np.mean(M2_calibration))

        LSM_calibration_times_N_estimation_sd.append(np.std(LSM_calibration))
        M1_calibration_times_N_estimation_sd.append(np.std(M1_calibration))
        M2_calibration_times_N_estimation_sd.append(np.std(M2_calibration))

        LSM_estimation_times_N_estimation.append(np.mean(LSM_estimation))
        M1_estimation_times_N_estimation.append(np.mean(M1_estimation))
        M2_estimation_times_N_estimation.append(np.mean(M2_estimation))

        LSM_estimation_times_N_estimation_sd.append(np.std(LSM_estimation))
        M1_estimation_times_N_estimation_sd.append(np.std(M1_estimation))
        M2_estimation_times_N_estimation_sd.append(np.std(M2_estimation))

    print(f"Varying estimation N:")
    print(f"backwards_pass_simulation_time_N_estimation = {backwards_pass_simulation_time_N_estimation}")
    print(f"forward_pass_simulation_time_N_estimation = {forward_pass_simulation_time_N_estimation}")

    print(f"LSM_calibration_times_N_estimation = {LSM_calibration_times_N_estimation}")
    print(f"M1_calibration_times_N_estimation = {M1_calibration_times_N_estimation}")
    print(f"M2_calibration_times_N_estimation = {M2_calibration_times_N_estimation}")
    print(f"LSM_estimation_times_N_estimation = {LSM_estimation_times_N_estimation}")
    print(f"M1_estimation_times_N_estimation = {M1_estimation_times_N_estimation}")
    print(f"M2_estimation_times_N_estimation = {M2_estimation_times_N_estimation}")

backwards_pass_simulation_time_T = [np.float64(0.018489210605621337),
                                    np.float64(0.05323436737060547),
                                    np.float64(0.10700492143630981)]

forward_pass_simulation_time_T = [np.float64(0.009496266841888428),
                                  np.float64(0.03022165298461914),
                                  np.float64(0.06200347423553467)]

LSM_calibration_times_T = [np.float64(0.02805861711502075),
                           np.float64(0.09551172256469727),
                           np.float64(0.1806342911720276)]

M1_calibration_times_T = [np.float64(0.03748821020126343),
                          np.float64(0.13705304622650147),
                          np.float64(0.2590860390663147)]

M2_calibration_times_T = [np.float64(0.031170415878295898),
                          np.float64(0.09273351907730103),
                          np.float64(0.1653588342666626)]

LSM_estimation_times_T = [np.float64(0.006364459991455078),
                          np.float64(0.015597219467163087),
                          np.float64(0.025175647735595705)]

M1_estimation_times_T = [np.float64(0.006108746528625488),
                         np.float64(0.015344514846801757),
                         np.float64(0.024644715785980223)]

M2_estimation_times_T = [np.float64(0.006090483665466309),
                         np.float64(0.015352423191070557),
                         np.float64(0.024655725955963135)]

backwards_pass_simulation_time_k = [np.float64(0.05319451093673706),
                                    np.float64(0.11602772235870361),
                                    np.float64(0.24493139505386352),
                                    np.float64(0.8019048762321472)]

forward_pass_simulation_time_k = [np.float64(0.030107414722442626),
                                  np.float64(0.06509207487106324),
                                  np.float64(0.13515065670013426),
                                  np.float64(0.424718382358551)]

LSM_calibration_times_k = [np.float64(0.09484692811965942),
                           np.float64(0.2383716082572937),
                           np.float64(0.5364459681510926),
                           np.float64(1.949227957725525)]

M1_calibration_times_k = [np.float64(0.13725085496902467),
                          np.float64(0.32272869348526),
                          np.float64(0.6996471881866455),
                          np.float64(2.187982888221741)]

M2_calibration_times_k = [np.float64(0.09228347539901734),
                          np.float64(0.21878024101257323),
                          np.float64(0.48089328289031985),
                          np.float64(1.6243583345413208)]

LSM_estimation_times_k = [np.float64(0.01570756673812866),
                          np.float64(0.03282464742660522),
                          np.float64(0.06845761060714722),
                          np.float64(0.20167813777923585)]

M1_estimation_times_k = [np.float64(0.015337183475494384),
                         np.float64(0.032418577671051024),
                         np.float64(0.06662938117980957),
                         np.float64(0.1982365894317627)]

M2_estimation_times_k = [np.float64(0.015375165939331055),
                         np.float64(0.03248633861541748),
                         np.float64(0.06694992303848267),
                         np.float64(0.20052754402160644)]

backwards_pass_simulation_time_N_calibration = [np.float64(0.028178791999816894),
                                                np.float64(0.10382211923599244),
                                                np.float64(0.06104107141494751),
                                                np.float64(0.3326126003265381)]

forward_pass_simulation_time_N_calibration = [np.float64(0.031042537689208984),
                                              np.float64(0.10459657907485961),
                                              np.float64(0.034719064235687255),
                                              np.float64(0.03366456985473633)]

LSM_calibration_times_N_calibration = [np.float64(0.094305579662323),
                                       np.float64(0.30711442708969117),
                                       np.float64(0.13000500440597534),
                                       np.float64(0.7776109743118286)]

M1_calibration_times_N_calibration = [np.float64(0.1021267318725586),
                                      np.float64(0.3225225758552551),
                                      np.float64(0.168664128780365),
                                      np.float64(1.2644002890586854)]

M2_calibration_times_N_calibration = [np.float64(0.06613265037536621),
                                      np.float64(0.23703495264053345),
                                      np.float64(0.10664097547531128),
                                      np.float64(0.9753905534744263)]

LSM_estimation_times_N_calibration = [np.float64(0.016073331832885743),
                                      np.float64(0.05359697341918945),
                                      np.float64(0.01747519016265869),
                                      np.float64(0.01735602378845215)]

M1_estimation_times_N_calibration = [np.float64(0.015817384719848632),
                                     np.float64(0.05198747634887695),
                                     np.float64(0.017431628704071046),
                                     np.float64(0.016994054317474364)]

M2_estimation_times_N_calibration = [np.float64(0.015831191539764405),
                                     np.float64(0.056014034748077396),
                                     np.float64(0.01725175142288208),
                                     np.float64(0.01669224739074707)]

backwards_pass_simulation_time_N_estimation = [np.float64(0.11226835489273071),
                                               np.float64(0.07075073719024658),
                                               np.float64(0.05322644710540771),
                                               np.float64(0.05946348428726196)]

forward_pass_simulation_time_N_estimation = [np.float64(0.06957784414291382),
                                             np.float64(0.03959631443023682),
                                             np.float64(0.053155148029327394),
                                             np.float64(0.3427420711517334)]

LSM_calibration_times_N_estimation = [np.float64(0.2529960298538208),
                                      np.float64(0.12472542524337768),
                                      np.float64(0.09608184576034545),
                                      np.float64(0.10369226932525635)]

M1_calibration_times_N_estimation = [np.float64(0.3096937680244446),
                                     np.float64(0.18786940813064576),
                                     np.float64(0.1390119504928589),
                                     np.float64(0.1527908205986023)]

M2_calibration_times_N_estimation = [np.float64(0.20917381048202516),
                                     np.float64(0.11372227907180786),
                                     np.float64(0.09381234169006347),
                                     np.float64(0.10718060493469238)]

LSM_estimation_times_N_estimation = [np.float64(0.03353160619735718),
                                     np.float64(0.023376598358154296),
                                     np.float64(0.02596069574356079),
                                     np.float64(0.14330225467681884)]

M1_estimation_times_N_estimation = [np.float64(0.04289272785186768),
                                    np.float64(0.019715273380279542),
                                    np.float64(0.025347058773040772),
                                    np.float64(0.13619880199432374)]

M2_estimation_times_N_estimation = [np.float64(0.03437623023986816),
                                    np.float64(0.018798198699951172),
                                    np.float64(0.025652661323547363),
                                    np.float64(0.13912498235702514)]