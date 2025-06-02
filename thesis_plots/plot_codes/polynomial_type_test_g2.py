from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel
import _config as cfg

from _LSM import LSM_method_v1

import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_eta
from thesis_plots.plot_codes._parameter_config import g2_rho

from thesis_plots.plot_codes._parameter_config import colors

def simulate_G2_results():
    N_calibration = 10000
    N_estimation = 1000

    instant_forward_rates = dict(zip(maturities, market_forward_rates))

    exercise_dates = [i*(T/M) for i in range(1, M+1) if i*(T/M) < T-alpha]

    GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho, instant_forward_rates=instant_forward_rates)

    for polynomial, _ in cfg.polynomial_classes.items():
        LSM = LSM_method_v1(strike=strike, exercise_dates=exercise_dates, basis_function=(polynomial, 3))

        LSM_IS_results = []
        LSM_OOS_results = []

        for i in range(100):
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=T, M=M, N=N_calibration,method='euler')
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T, M=M, N=N_estimation,method='euler')

            swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                                  y_paths=y_calibration,
                                                                                                  varphi=varphi_calibration,
                                                                                                  entry_dates=exercise_dates,
                                                                                                  expiry=T,
                                                                                                  alpha=alpha)

            swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                                y_paths=y_estimation,
                                                                                                varphi=varphi_estimation,
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
    'power': 0.11545891549661523,
    'chebyshev': 0.11544694398988391,
    'legendre': 0.1154595565104981,
    'laguerre': 0.11544529629752519,
    'hermite': 0.11540043915076835
}

IS_sd = {
    'power': 0.00037271530533076824,
    'chebyshev': 0.0004004847170165181,
    'legendre': 0.00032559354180964294,
    'laguerre': 0.0003687597879294106,
    'hermite': 0.00033698467069638306
}

OOS_means = {
    'power': 0.11512385229855314,
    'chebyshev': 0.11543514168253335,
    'legendre': 0.11572844012362117,
    'laguerre': 0.1153434885394519,
    'hermite': 0.11560302859712333
}

OOS_sd = {
    'power': 0.0009513068930234254,
    'chebyshev': 0.0010949037628964176,
    'legendre': 0.001063145494621927,
    'laguerre': 0.0009994901153076614,
    'hermite': 0.0010854048536344258
}

IS_means_lst = [IS_means[key] for key, _ in IS_means.items()]
OOS_means_lst = [OOS_means[key] for key, _ in OOS_means.items()]

barWidth = 0.25
r = np.arange(len(IS_means_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_means_lst, color=colors['dark_green'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_means_lst, color=colors['light_green'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels(['Power', 'Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_type_test_mean_g2.png',
            bbox_inches='tight')
plt.show()

IS_sd_lst = [IS_sd[key] for key, _ in IS_sd.items()]
OOS_sd_lst = [OOS_sd[key] for key, _ in OOS_sd.items()]

barWidth = 0.25
r = np.arange(len(IS_sd_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_sd_lst, color=colors['dark_green'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_sd_lst, color=colors['light_green'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels(['Power', 'Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_type_test_sd_g2.png',
            bbox_inches='tight')
plt.show()