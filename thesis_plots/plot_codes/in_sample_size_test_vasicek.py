from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import vasicek_a
from thesis_plots.plot_codes._parameter_config import vasicek_b
from thesis_plots.plot_codes._parameter_config import vasicek_sigma
from thesis_plots.plot_codes._parameter_config import vasicek_r_0

from thesis_plots.plot_codes._parameter_config import colors

def simulate_data():
    exercise_dates = [i * (T / M) for i in range(1, M + 1) if i * (T / M) < T - alpha]

    VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

    for n in [100, 1000, 5000, 10000, 20000, 50000, 100000]:
        LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

        LSM_IS_results = []
        LSM_OOS_results = []

        M1_IS_results = []
        M1_OOS_results = []

        M2_IS_results = []
        M2_OOS_results = []

        for i in range(100):
            short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=n, method='exact')
            short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=T, M=M, N=1000, method='exact')

            swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
                short_rate=short_rates_calibration,
                entry_dates=exercise_dates,
                expiry=T,
                alpha=alpha)

            swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
                short_rate=short_rates_estimation,
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

            M2_IS, M2_betas = LSM.calibration(method='short_rate_delta',
                                              underlying_asset_paths=swap_rates_calibration.copy(),
                                              payoffs=calibration_payoffs.copy(),
                                              discount_factors=discount_factors_calibration.copy(),
                                              a=vasicek_a)

            LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                     payoffs=estimation_payoffs.copy(),
                                     discount_factors=discount_factors_estimation.copy(),
                                     betas=LSM_betas.copy())

            M1_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M1_betas.copy())

            M2_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                                    payoffs=estimation_payoffs.copy(),
                                    discount_factors=discount_factors_estimation.copy(),
                                    betas=M2_betas.copy())

            LSM_IS_results.append(LSM_IS)
            LSM_OOS_results.append(LSM_OOS)
            M1_IS_results.append(M1_IS)
            M1_OOS_results.append(M1_OOS)
            M2_IS_results.append(M2_IS)
            M2_OOS_results.append(M2_OOS)

        LSM_IS_mean = np.mean(LSM_IS_results)
        LSM_IS_sd = np.std(LSM_IS_results)
        LSM_OOS_mean = np.mean(LSM_OOS_results)
        LSM_OOS_sd = np.std(LSM_OOS_results)

        M1_IS_mean = np.mean(M1_IS_results)
        M1_IS_sd = np.std(M1_IS_results)
        M1_OOS_mean = np.mean(M1_OOS_results)
        M1_OOS_sd = np.std(M1_OOS_results)

        M2_IS_mean = np.mean(M2_IS_results)
        M2_IS_sd = np.std(M2_IS_results)
        M2_OOS_mean = np.mean(M2_OOS_results)
        M2_OOS_sd = np.std(M2_OOS_results)

        print(f"Using {n} In-sample paths and 1000 Out-of-sample paths:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")
        print(f"M1 In-sample: {M1_IS_mean}, M1 Out-of-sample: {M1_OOS_mean}")
        print(f"M1 In-sample sd: {M1_IS_sd}, M1 Out-of-sample sd: {M1_OOS_sd}")
        print(f"M2 In-sample: {M2_IS_mean}, M2 Out-of-sample: {M2_OOS_mean}")
        print(f"M2 In-sample sd: {M2_IS_sd}, M2 Out-of-sample sd: {M2_OOS_sd}")

#Results as of 24/05/2025
LSM_IS_result = [
    0.11446122974857299,
    0.1153348412448423,
    0.11501758174228026,
    0.1149974329050969,
    0.11508015310773281,
    0.11504681968182182,
    0.11506406687888454
]

M1_IS_result = [
    0.11278625695269995,
    0.11407568993345331,
    0.11371915261131665,
    0.11379233924055657,
    0.11382630747697858,
    0.11379841706478841,
    0.11381902328471206
]

M2_IS_result = [
    0.11341824456352297,
    0.11454628047357994,
    0.1142077116588511,
    0.11425276732791223,
    0.1143106322553454,
    0.11427616321308796,
    0.11430251419301701
]

LSM_OOS_result = [
    0.11430297392336261,
    0.11484370451580869,
    0.11512187879826609,
    0.1148440066811145,
    0.11515422430283451,
    0.11509081706842775,
    0.11505052663029595
]

M1_OOS_result = [
    0.11379683253287498,
    0.11375693908300809,
    0.11390607268976233,
    0.11354003636774371,
    0.11392232999391481,
    0.11380887612151541,
    0.11377788161706849
]

M2_OOS_result = [
    0.11425173956212531,
    0.11423569313559209,
    0.11437196943255362,
    0.11405156765941495,
    0.11439982426383401,
    0.1142566028828472,
    0.11423736320723765
]

LSM_IS_sd = [
    0.004706146766107353,
    0.0015240108326300297,
    0.0006704510835791863,
    0.00048803282658080235,
    0.00036115888769294296,
    0.0002404291725131615,
    0.00016178802380268167
]

M1_IS_sd = [
    0.0050270496812623655,
    0.0016075120636376175,
    0.0006802932298949989,
    0.0005148989756828032,
    0.0003483193674132612,
    0.00023910218294989532,
    0.00016381880987146116
]

M2_IS_sd = [
    0.005074345247652984,
    0.001632580417739387,
    0.0006947767981958661,
    0.0005135206271409868,
    0.00035167187872948486,
    0.00024497882390000396,
    0.0001626451914687437
]

LSM_OOS_sd = [
    0.0013959201325211997,
    0.0016711710260398289,
    0.0016699351311902438,
    0.0015785368093940249,
    0.001662838528624625,
    0.0016729054560885397,
    0.0017147900323043164
]

M1_OOS_sd = [
    0.0013681805547616204,
    0.0017029873979144852,
    0.0018672812691380245,
    0.0015327238680555861,
    0.0018254634033221782,
    0.0016671865749828074,
    0.0017740963044346751
]

M2_OOS_sd = [
    0.0013279239756786598,
    0.0016853121349976014,
    0.00180061368323261,
    0.001521782157442022,
    0.0017929945812860458,
    0.0017075670428901954,
    0.0017310364681090826
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
ax.errorbar(N_numeric - offset, LSM_IS_result, yerr=LSM_IS_sd, fmt='o', color=colors['dark_blue'], label='LSM', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M1_IS_result with a slight offset
ax.errorbar(N_numeric, M1_IS_result, yerr=M1_IS_sd, fmt='o', color=colors['light_petrol'], label='Delta M1', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M2_IS_result with a slight offset
ax.errorbar(N_numeric + offset, M2_IS_result, yerr=M2_IS_sd, fmt='o', color=colors['medium_petrol'], label='Delta M2', markersize=8, capsize=5, elinewidth=1, ecolor='black')

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
           markerfacecolor=colors['dark_blue'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Delta M1',
           markerfacecolor=colors['light_petrol'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Delta M2',
           markerfacecolor=colors['medium_petrol'], markersize=8)
]

# Add the custom legend
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

# Adjust layout and show
plt.tight_layout()
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\IS_IS_test_vasicek.png',
       bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

# Convert N to string for categorical x-axis
N_labels = [str(n) for n in N]

# Define the width for the offset within each category
offset = 0.2  # Adjust this to control the spacing between the results

# Create positions for each group within the same category
N_numeric = np.arange(len(N))  # These are the positions for the categories on the x-axis

# Plot LSM_IS_result with a slight offset#343a40
ax.errorbar(N_numeric - offset, LSM_OOS_result, yerr=LSM_OOS_sd, fmt='o', color=colors['dark_blue'], label='LSM', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M1_IS_result with a slight offset
ax.errorbar(N_numeric, M1_OOS_result, yerr=M1_OOS_sd, fmt='o', color=colors['light_petrol'], label='Delta M1', markersize=8, capsize=5, elinewidth=1, ecolor='black')

# Plot M2_IS_result with a slight offset
ax.errorbar(N_numeric + offset, M2_OOS_result, yerr=M2_OOS_sd, fmt='o', color=colors['medium_petrol'], label='Delta M2', markersize=8, capsize=5, elinewidth=1, ecolor='black')

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
           markerfacecolor=colors['dark_blue'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Delta M1',
           markerfacecolor=colors['light_petrol'], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Delta M2',
           markerfacecolor=colors['medium_petrol'], markersize=8)
]

# Add the custom legend
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

# Adjust layout and show
plt.tight_layout()
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\IS_OOS_test_vasicek.png',
       bbox_inches='tight')
plt.show()
