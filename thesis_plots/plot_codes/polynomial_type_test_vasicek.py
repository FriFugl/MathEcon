from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel
import _config as cfg

from _LSM import LSM_method_v1

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

def simulate_results():
    N_calibration = 10000
    N_estimation = 1000

    exercise_dates = [i*(T/M) for i in range(1, M+1) if i*(T/M) < T-alpha]

    VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

    for polynomial, _ in cfg.polynomial_classes.items():
        LSM = LSM_method_v1(strike=strike, exercise_dates=exercise_dates, basis_function=(polynomial, 3))

        LSM_IS_results = []
        LSM_OOS_results = []

        for i in range(100):
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_calibration,method='exact')
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=N_estimation,method='exact')

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

            LSM_IS, LSM_betas = LSM.calibration(underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     fitted_basis_functions=LSM_betas.copy())

            LSM_IS_results.append(LSM_IS)
            LSM_OOS_results.append(LSM_OOS)

        LSM_IS_mean = np.mean(LSM_IS_results)
        LSM_IS_sd = np.std(LSM_IS_results)

        LSM_OOS_mean = np.mean(LSM_OOS_results)
        LSM_OOS_sd = np.std(LSM_OOS_results)

        print(f"Using {polynomial} polynomials of degree 3:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")

#Results as of 24/05/2025
IS_means = {
    'power': 0.11502478872864756,
    'chebyshev': 0.1151055199794783,
    'legendre': 0.11510283163224992,
    'laguerre': 0.11495287528158818,
    'hermite': 0.11500234944747766
}

IS_sd = {
    'power': 0.0004938906325883081,
    'chebyshev': 0.0004632594286586993,
    'legendre': 0.0006113544462033371,
    'laguerre': 0.0005495773575473395,
    'hermite': 0.00046897430070550524
}

OOS_means = {
    'power': 0.11500081499413406,
    'chebyshev': 0.11524144295870259,
    'legendre': 0.11516376499189702,
    'laguerre': 0.11499619939043346,
    'hermite': 0.1151106986007109
}

OOS_sd = {
    'power': 0.001722533809294219,
    'chebyshev': 0.001684378140142276,
    'legendre': 0.0015931269555848051,
    'laguerre': 0.0015165069275673436,
    'hermite': 0.0016297050104155319
}

IS_means_lst = [IS_means[key] for key, _ in IS_means.items()]
OOS_means_lst = [OOS_means[key] for key, _ in OOS_means.items()]

barWidth = 0.25
r = np.arange(len(IS_means_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_means_lst, color=colors['dark_blue'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_means_lst, color=colors['light_petrol'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels(['Power', 'Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_type_test_mean_vasicek.png',
            bbox_inches='tight')
plt.show()

IS_sd_lst = [IS_sd[key] for key, _ in IS_sd.items()]
OOS_sd_lst = [OOS_sd[key] for key, _ in OOS_sd.items()]

barWidth = 0.25
r = np.arange(len(IS_sd_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_sd_lst, color=colors['dark_blue'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_sd_lst, color=colors['light_petrol'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels(['Power', 'Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_type_test_sd_vasicek.png',
            bbox_inches='tight')
plt.show()


