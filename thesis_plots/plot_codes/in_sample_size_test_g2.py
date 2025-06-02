from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
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
            short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=T, M=M, N=n, method='euler')
            short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=T, M=M, N=1000, method='euler')

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

        print(f"Using {n} In-sample paths and 1000 Out-of-sample paths:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")
        print(f"M1 In-sample: {M1_IS_mean}, M1 Out-of-sample: {M1_OOS_mean}")
        print(f"M1 In-sample sd: {M1_IS_sd}, M1 Out-of-sample sd: {M1_OOS_sd}")

#Results as of 24/05/2025
LSM_IS_result = [
    0.11569954233492061,
    0.11547637608488003,
    0.11541259478872369,
    0.11544719152236865,
    0.1154299786807712,
    0.11542659207505103,
    0.11541813908214824
]

M1_IS_result = [
    0.11426320445582705,
    0.1141345778387454,
    0.11413200781938906,
    0.1141483673929838,
    0.11412561236007289,
    0.11413132647656779,
    0.11413667373744207
]

LSM_OOS_result = [
    0.1147842007717802,
    0.1155879497581796,
    0.1153029605938183,
    0.11538919713799423,
    0.11539530997926555,
    0.1155893723135347,
    0.11547623112457732
]

M1_OOS_result = [
    0.11413810529256015,
    0.11431786881597686,
    0.11407717918129061,
    0.11405950456364228,
    0.11405144723606064,
    0.11429094200676712,
    0.11422267109267642
]

LSM_IS_sd = [
    0.0034749321672723067,
    0.0011887183736824251,
    0.0005172732955982027,
    0.0003522461038427291,
    0.0002620311286721198,
    0.00015064368845374809,
    0.00012374721661357428
]

M1_IS_sd = [
    0.0036432743623652335,
    0.001153399396811291,
    0.000535112452209924,
    0.0003875713308212807,
    0.0002750567407180094,
    0.00016856789957872583,
    0.00013385583996612964
]

LSM_OOS_sd = [
    0.0013324378676090448,
    0.0012214808981833429,
    0.001156767513005317,
    0.0011335420032349487,
    0.0010020663772387574,
    0.00120802348315053,
    0.0010920414515942043
]

M1_OOS_sd = [
    0.0013066731916422062,
    0.0013845693731474294,
    0.0013033691325065732,
    0.0011799589167469733,
    0.001105203746382001,
    0.0013199411639399107,
    0.0013659590939277255
]

N = [100, 1000, 5000, 10000, 20000, 50000, 100000]
N_labels = [str(n) for n in N]

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

# Convert N to string for categorical x-axis
N_labels = [str(n) for n in N]

# Define the width for the offset within each category
offset = 0.1  # Adjust this to control the spacing between the results

# Create positions for each group within the same category
N_numeric = np.arange(len(N))  # These are the positions for the categories on the x-axis

# Plot LSM_IS_result with a slight offset#343a40
ax.errorbar(N_numeric - offset, LSM_IS_result, yerr=LSM_IS_sd, fmt='o', color=colors['dark_green'], label='LSM', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M1_IS_result with a slight offset
ax.errorbar(N_numeric + offset, M1_IS_result, yerr=M1_IS_sd, fmt='o', color=colors['light_green'], label='Delta M1', markersize=8, capsize=5, elinewidth=1, ecolor='black')
# Customize x-ticks to display categorical values (N)
ax.set_xticks(N_numeric)
ax.set_xticklabels(N_labels)


# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add labels and title
plt.xlabel('Number of In-sample simulations')
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
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\IS_IS_test_g2.png',
           bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

# Convert N to string for categorical x-axis
N_labels = [str(n) for n in N]

# Define the width for the offset within each category
offset = 0.1  # Adjust this to control the spacing between the results

# Create positions for each group within the same category
N_numeric = np.arange(len(N))  # These are the positions for the categories on the x-axis

# Plot LSM_IS_result with a slight offset#343a40
ax.errorbar(N_numeric - offset, LSM_OOS_result, yerr=LSM_OOS_sd, fmt='o', color=colors['dark_green'], label='LSM', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M1_IS_result with a slight offset
ax.errorbar(N_numeric + offset, M1_OOS_result, yerr=M1_OOS_sd, fmt='o', color=colors['light_green'], label='Delta M1', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Customize x-ticks to display categorical values (N)
ax.set_xticks(N_numeric)
ax.set_xticklabels(N_labels)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add labels and title
plt.xlabel('Number of In-sample simulations')
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
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\IS_OOS_test_g2.png',
           bbox_inches='tight')
plt.show()

