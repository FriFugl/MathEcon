from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_eta
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_rho
from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates

from thesis_plots.plot_codes._parameter_config import colors

instant_forward_rates = dict(zip(maturities, market_forward_rates))


exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

def simulate_data():
    GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                          instant_forward_rates=instant_forward_rates)

    for n in [100, 1000, 5000, 10000, 20000, 50000, 100000]:
        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        LSM_IS_results = []
        LSM_OOS_results = []

        M1_IS_results = []
        M1_OOS_results = []

        for i in range(100):
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=T, M=M, N=10000, method='euler')
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T, M=M, N=n, method='euler')

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

            calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                              accrual_factors=accrual_factors_calibration, strike=strike)
            estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                             accrual_factors=accrual_factors_estimation, strike=strike)

            discount_factors_calibration = _short_rate_to_discount_factors(short_rates=short_rates_calibration)
            discount_factors_estimation = _short_rate_to_discount_factors(short_rates=short_rates_estimation)

            LSM_IS, LSM_betas = LSM.calibration(method='classic',
                                                underlying_asset_paths=swap_rates_calibration.copy(),
                                                payoffs=calibration_payoffs.copy(),
                                                discount_factors=discount_factors_calibration.copy())

            M1_IS, M1_betas = LSM.calibration(method='swap_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              accrual_factors=accrual_factors_calibration.copy())

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())

            LSM_IS_results.append(LSM_IS)
            LSM_OOS_results.append(LSM_OOS)
            M1_IS_results.append(M1_IS)
            M1_OOS_results.append(M1_OOS)

        LSM_IS_mean = np.mean(LSM_IS_results)
        LSM_IS_sd = np.std(LSM_IS_results)
        LSM_OOS_mean = np.mean(LSM_OOS_results)
        LSM_OOS_sd = np.std(LSM_OOS_results)

        M1_IS_mean = np.mean(M1_IS_results)
        M1_IS_sd = np.std(M1_IS_results)
        M1_OOS_mean = np.mean(M1_OOS_results)
        M1_OOS_sd = np.std(M1_OOS_results)

        print(f"Using 10000 In-sample paths and {n} Out-of-sample paths:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")
        print(f"M1 In-sample: {M1_IS_mean}, M1 Out-of-sample: {M1_OOS_mean}")
        print(f"M1 In-sample sd: {M1_IS_sd}, M1 Out-of-sample sd: {M1_OOS_sd}")

#Result as of 24/05/2025
LSM_IS_result = [
    0.11549753431340229,
    0.1154749572421329,
    0.11543887094586189,
    0.11546638542057563,
    0.11543632727880236,
    0.11544683185607761,
    0.11545386517625063
]

M1_IS_result = [
    0.11420017945785162,
    0.11421936794232035,
    0.11413941925012738,
    0.11418454530330684,
    0.11415724362198608,
    0.11416319706960565,
    0.11416495262824554
]

LSM_OOS_result = [
    0.11562648747209389,
    0.11544470893135599,
    0.11537525779886247,
    0.11538998547631571,
    0.1154348276591166,
    0.11545058037943515,
    0.11542859035788283
]

M1_OOS_result = [
    0.11416589932313133,
    0.11417802300465009,
    0.11409421263296375,
    0.11411516289261957,
    0.11415554526268207,
    0.11418597707289299,
    0.11415795113843777
]

LSM_IS_sd = [
    0.0003677475329204233,
    0.00036978387499921213,
    0.00037080359239335595,
    0.0003513651636687146,
    0.00035815079793097564,
    0.00034353527695670313,
    0.00038807079148261455
]

M1_IS_sd = [
    0.00037550834140502636,
    0.000383331299014131,
    0.000364068724252955,
    0.0003615832058758968,
    0.0003597060024532846,
    0.00036841049154605386,
    0.0003852656771919055
]

LSM_OOS_sd = [
    0.003427044100308692,
    0.001017729336643969,
    0.0005369991099270989,
    0.0003273616172604472,
    0.00028045199789289253,
    0.00016766670539409138,
    0.00010934584974639605
]

M1_OOS_sd = [
    0.003755733202998224,
    0.0011728724023513403,
    0.0005995928932938225,
    0.00037362964013212433,
    0.0003070394333319407,
    0.00018765463830767683,
    0.00013223821465336124
]

N = [100, 1000, 5000, 10000, 20000, 50000, 100000]
N_labels = [str(n) for n in N]

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

# Convert N to string for categorical x-axis
N_labels = [str(n) for n in N]

# Define the width for the offset within each category
offset = 0.2  # Adjust this to control the spacing between the results

# Create positions for each group within the same category
N_numeric = np.arange(len(N))  # These are the positions for the categories on the x-axis

# Plot LSM_IS_result with a slight offset#343a40
ax.errorbar(N_numeric - offset, LSM_OOS_result, yerr=LSM_OOS_sd, fmt='o', color=colors['dark_green'], label='LSM', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M1_IS_result with a slight offset
ax.errorbar(N_numeric, M1_OOS_result, yerr=M1_OOS_sd, fmt='o', color=colors['light_green'], label='Delta M1', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Customize x-ticks to display categorical values (N)
ax.set_xticks(N_numeric)
ax.set_xticklabels(N_labels)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add labels and title
plt.xlabel('Number of Out-of-sample simulations')
plt.ylabel('Price estimates')

# Add a grid for readability
plt.grid(axis='y', alpha=0.7)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='LSM',
           markerfacecolor=colors['dark_green'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Delta M1',
           markerfacecolor=colors['light_green'], markersize=8)
]

# Add the custom legend
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

# Adjust layout and show
plt.tight_layout()
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\OOS_OOS_test_g2.png',
          bbox_inches='tight')
plt.show()
