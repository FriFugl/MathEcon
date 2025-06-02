from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import vasicek_a
from thesis_plots.plot_codes._parameter_config import vasicek_b
from thesis_plots.plot_codes._parameter_config import vasicek_sigma
from thesis_plots.plot_codes._parameter_config import vasicek_r_0

from thesis_plots.plot_codes._parameter_config import colors


def simulate_vasicek_results():
    N_calibration = 10000
    N_estimation = 1000

    exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

    VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

    for degree in [i for i in range(1, 11)]:
        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=degree)

        LSM_IS_results = []
        LSM_OOS_results = []

        for i in range(100):
            short_rates_calibration = VasicekModelInstance.simulate(
                r_0=vasicek_r_0, T=T, M=M, N=N_calibration, method="exact"
            )
            short_rates_estimation = VasicekModelInstance.simulate(
                r_0=vasicek_r_0, T=T, M=M, N=N_estimation, method="exact"
            )

            swap_rates_calibration, accrual_factors_calibration = (
                VasicekModelInstance.swap_rate(
                    short_rate=short_rates_calibration,
                    entry_dates=exercise_dates,
                    expiry=T,
                    alpha=alpha,
                )
            )

            swap_rates_estimation, accrual_factors_estimation = (
                VasicekModelInstance.swap_rate(
                    short_rate=short_rates_estimation,
                    entry_dates=exercise_dates,
                    expiry=T,
                    alpha=alpha,
                )
            )

            calibration_payoffs = _calculate_swaption_payoffs(
                swap_rates=swap_rates_calibration,
                accrual_factors=accrual_factors_calibration,
                strike=strike,
            )
            estimation_payoffs = _calculate_swaption_payoffs(
                swap_rates=swap_rates_estimation,
                accrual_factors=accrual_factors_estimation,
                strike=strike,
            )

            discount_factors_calibration = _short_rate_to_discount_factors(
                short_rates=short_rates_calibration
            )
            discount_factors_estimation = _short_rate_to_discount_factors(
                short_rates=short_rates_estimation
            )

            LSM_IS, LSM_betas = LSM.calibration(
                method="classic",
                underlying_asset_paths=swap_rates_calibration.copy(),
                payoffs=calibration_payoffs.copy(),
                discount_factors=discount_factors_calibration.copy(),
            )

            LSM_OOS = LSM.estimation(
                underlying_asset_paths=swap_rates_estimation.copy(),
                payoffs=estimation_payoffs.copy(),
                discount_factors=discount_factors_estimation.copy(),
                betas=LSM_betas.copy(),
            )

            LSM_IS_results.append(LSM_IS)
            LSM_OOS_results.append(LSM_OOS)

        LSM_IS_mean = np.mean(LSM_IS_results)
        LSM_IS_sd = np.std(LSM_IS_results)
        LSM_OOS_mean = np.mean(LSM_OOS_results)
        LSM_OOS_sd = np.std(LSM_OOS_results)

        print(f"Using polynomials of degree {degree}:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")


# Results as of 24/05/2025
LSM_IS_means = {
    1: 0.11493790136180757,
    2: 0.11496149505240659,
    3: 0.11505739783238583,
    4: 0.11508191436447301,
    5: 0.11497179129408171,
    6: 0.11509364041288729,
    7: 0.11505583395780693,
    8: 0.1150911038580038,
    9: 0.1151384358654683,
    10: 0.11507952952887134,
}

LSM_IS_sd = {
    1: 0.0005134843987371503,
    2: 0.0005205800578408848,
    3: 0.0005690267600772523,
    4: 0.00047681783458682867,
    5: 0.0005534468784762823,
    6: 0.0005468115405703792,
    7: 0.0005605934705007209,
    8: 0.0005516489864503843,
    9: 0.0005001082612359057,
    10: 0.00046264212126244776,
}

LSM_OOS_means = {
    1: 0.11516120894107554,
    2: 0.114822246677717,
    3: 0.11504237998760736,
    4: 0.11511457428686084,
    5: 0.11500202048531909,
    6: 0.11512693527488016,
    7: 0.11477116859189222,
    8: 0.11478162828835219,
    9: 0.114799821862537,
    10: 0.11513280881876506,
}

LSM_OOS_sd = {
    1: 0.001954142799134418,
    2: 0.001733359254570279,
    3: 0.001539702969431226,
    4: 0.0016224783072876392,
    5: 0.001753866204770859,
    6: 0.0015460249887046462,
    7: 0.001787740694871307,
    8: 0.001709468890657939,
    9: 0.0015087954048953792,
    10: 0.001619770213438336,
}

IS_means_lst = [LSM_IS_means[key] for key, _ in LSM_IS_means.items()]
OOS_means_lst = [LSM_OOS_means[key] for key, _ in LSM_OOS_means.items()]

IS_sd_lst = [LSM_IS_sd[key] for key, _ in LSM_IS_sd.items()]
OOS_sd_lst = [LSM_OOS_sd[key] for key, _ in LSM_OOS_sd.items()]

barWidth = 0.25
r = np.arange(len(LSM_IS_means))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
ax.bar(
    r,
    IS_means_lst,
    color=colors["dark_blue"],
    width=barWidth,
    edgecolor="white",
    label="Backward pass",
)
ax.bar(
    r2,
    OOS_means_lst,
    color=colors["light_petrol"],
    width=barWidth,
    edgecolor="white",
    label="Forward pass",
)

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels([i for i in range(1, 11)])
ax.tick_params(axis="x", which="both", bottom=False, top=False)

# Legend and show
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_degree_test_mean_vasicek.png",
    bbox_inches="tight",
)
plt.show()

barWidth = 0.25
r = np.arange(len(LSM_IS_sd))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
ax.bar(
    r,
    IS_sd_lst,
    color=colors["dark_blue"],
    width=barWidth,
    edgecolor="white",
    label="Backward pass",
)
ax.bar(
    r2,
    OOS_sd_lst,
    color=colors["light_petrol"],
    width=barWidth,
    edgecolor="white",
    label="Forward pass",
)

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels([i for i in range(1, 11)])
ax.tick_params(axis="x", which="both", bottom=False, top=False)

# Legend and show
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_degree_test_sd_vasicek.png",
    bbox_inches="tight",
)
plt.show()
