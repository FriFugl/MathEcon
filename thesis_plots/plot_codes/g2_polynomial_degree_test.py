from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel

from _LSM import LSM_method_v2

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

    GaussianModelInstance = GaussianModel(a=g2_a,
                                          b=g2_b,
                                          sigma=g2_sigma,
                                          eta=g2_eta,
                                          rho=g2_rho,
                                          instant_forward_rates=instant_forward_rates)

    for degree in [i for i in range(1,11)]:
        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=degree)

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

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            LSM_IS_results.append(LSM_IS)
            LSM_OOS_results.append(LSM_OOS)

        LSM_IS_mean = np.mean(LSM_IS_results)
        LSM_IS_sd = np.std(LSM_IS_results)
        LSM_OOS_mean = np.mean(LSM_OOS_results)
        LSM_OOS_sd = np.std(LSM_OOS_results)

        print(f"Using polynomials of degree {degree}:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")

#Results as of 24/05/2025
LSM_IS_means = {
    1: 0.11536463691477662,
    2: 0.11543233051279449,
    3: 0.11548474372538545,
    4: 0.11545582471523234,
    5: 0.11544892112964186,
    6: 0.11545760118725001,
    7: 0.11543781624053656,
    8: 0.1153769493419082,
    9: 0.11545615596085879,
    10: 0.11529999607263722
}

LSM_IS_sd = {
    1: 0.0003702809813462366,
    2: 0.00037102743060322404,
    3: 0.00034033101503945815,
    4: 0.000308922731605472,
    5: 0.00037393071415711405,
    6: 0.0003960623290055023,
    7: 0.00043106990295848637,
    8: 0.00032149509204212114,
    9: 0.0003338302173355624,
    10: 0.0003693615347982099
}

LSM_OOS_means = {
    1: 0.11551657760018673,
    2: 0.1154034304362691,
    3: 0.1154294909798093,
    4: 0.1155769570134087,
    5: 0.11539887929617447,
    6: 0.11547215512291834,
    7: 0.11545265254111302,
    8: 0.11540827379089358,
    9: 0.11560295589083855,
    10: 0.11537112099287448
}

LSM_OOS_sd = {
    1: 0.0011700037419807507,
    2: 0.001150417026705247,
    3: 0.001144320624230792,
    4: 0.0011663209437382794,
    5: 0.0010362330087538253,
    6: 0.00104670403807371,
    7: 0.0011712901366535962,
    8: 0.001249582636737683,
    9: 0.0010561881637426455,
    10: 0.0012122355298354057
}

IS_means_lst = [LSM_IS_means[key] for key, _ in LSM_IS_means.items()]
OOS_means_lst = [LSM_OOS_means[key] for key, _ in LSM_OOS_means.items()]

IS_sd_lst = [LSM_IS_sd[key] for key, _ in LSM_IS_sd.items()]
OOS_sd_lst = [LSM_OOS_sd[key] for key, _ in LSM_OOS_sd.items()]

barWidth = 0.25
r = np.arange(len(IS_means_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_means_lst, color=colors['dark_green'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_means_lst, color=colors['light_green'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels([i for i in range(1,11)])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_degree_test_mean_g2.png',
            bbox_inches='tight')
plt.show()

barWidth = 0.25
r = np.arange(len(IS_sd_lst))
r2 = r + barWidth

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
ax.bar(r, IS_sd_lst, color=colors['dark_green'], width=barWidth, edgecolor='white', label='Backward pass')
ax.bar(r2, OOS_sd_lst, color=colors['light_green'], width=barWidth, edgecolor='white', label='Forward pass')

# Xticks
ax.set_xticks(r + barWidth / 2)
ax.set_xticklabels([i for i in range(1,11)])
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Legend and show
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\polynomial_degree_test_sd_g2.png',
            bbox_inches='tight')
plt.show()