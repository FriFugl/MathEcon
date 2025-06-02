from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel

from _LSM import LSM_method_v2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
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
            short_rates_calibration = VasicekModelInstance.simulate(
                r_0=vasicek_r_0, T=T, M=M, N=10000, method="exact"
            )
            short_rates_estimation = VasicekModelInstance.simulate(
                r_0=vasicek_r_0, T=T, M=M, N=n, method="exact"
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

            M1_IS, M1_betas = LSM.calibration(
                method="swap_delta",
                underlying_asset_paths=swap_rates_calibration.copy(),
                payoffs=calibration_payoffs.copy(),
                discount_factors=discount_factors_calibration.copy(),
                accrual_factors=accrual_factors_calibration.copy(),
            )

            M2_IS, M2_betas = LSM.calibration(
                method="short_rate_delta",
                underlying_asset_paths=swap_rates_calibration.copy(),
                payoffs=calibration_payoffs.copy(),
                discount_factors=discount_factors_calibration.copy(),
                a=vasicek_a,
            )

            LSM_OOS = LSM.estimation(
                underlying_asset_paths=swap_rates_estimation.copy(),
                payoffs=estimation_payoffs.copy(),
                discount_factors=discount_factors_estimation.copy(),
                betas=LSM_betas.copy(),
            )

            M1_OOS = LSM.estimation(
                underlying_asset_paths=swap_rates_estimation.copy(),
                payoffs=estimation_payoffs.copy(),
                discount_factors=discount_factors_estimation.copy(),
                betas=M1_betas.copy(),
            )

            M2_OOS = LSM.estimation(
                underlying_asset_paths=swap_rates_estimation.copy(),
                payoffs=estimation_payoffs.copy(),
                discount_factors=discount_factors_estimation.copy(),
                betas=M2_betas.copy(),
            )

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

        print(f"Using 10000 In-sample paths and {n} Out-of-sample paths:")
        print(f"LSM In-sample: {LSM_IS_mean}, LSM Out-of-sample: {LSM_OOS_mean}")
        print(f"LSM In-sample sd: {LSM_IS_sd}, LSM Out-of-sample sd: {LSM_OOS_sd}")
        print(f"M1 In-sample: {M1_IS_mean}, M1 Out-of-sample: {M1_OOS_mean}")
        print(f"M1 In-sample sd: {M1_IS_sd}, M1 Out-of-sample sd: {M1_OOS_sd}")
        print(f"M2 In-sample: {M2_IS_mean}, M2 Out-of-sample: {M2_OOS_mean}")
        print(f"M2 In-sample sd: {M2_IS_sd}, M2 Out-of-sample sd: {M2_OOS_sd}")


# 24/05/2025
LSM_IS_result = [
    0.11507445228229679,
    0.11502646489298567,
    0.11507168707862313,
    0.11509080188044528,
    0.1150744829698183,
    0.11505923576503846,
    0.11506687260594217,
]

M1_IS_result = [
    0.11380439869110463,
    0.11373011029852603,
    0.11382024378111294,
    0.11386336173859467,
    0.113784205281173,
    0.11378986828440567,
    0.11379663065345484,
]

M2_IS_result = [
    0.11429372186115412,
    0.11422492779617524,
    0.11430896663898789,
    0.11432908721076604,
    0.11426554345568908,
    0.11426554043179561,
    0.11427718995165177,
]

LSM_OOS_result = [
    0.11552552480284167,
    0.11517912787344187,
    0.11502491014056096,
    0.11499160224318798,
    0.11498982054046528,
    0.11505926286104408,
    0.11505349302371835,
]

M1_OOS_result = [
    0.11422376109152131,
    0.11387148044263327,
    0.11379326838511697,
    0.11373788489785869,
    0.11378100269507904,
    0.11383284987304763,
    0.11380291690793483,
]

M2_OOS_result = [
    0.11471496448252488,
    0.1143055435520072,
    0.11426640340455829,
    0.11423627492789405,
    0.11425069012835831,
    0.11430671921413531,
    0.11428891086040201,
]

LSM_IS_sd = [
    0.0005197428249200265,
    0.0005155564497352853,
    0.0005393827746574552,
    0.0005429899190783347,
    0.0005810965594849699,
    0.0005600405318231739,
    0.0005351787047420741,
]

M1_IS_sd = [
    0.0005349747717090034,
    0.0005274072193630775,
    0.0005295290390227105,
    0.0005507505413203368,
    0.0005798385473254876,
    0.0005681551467749413,
    0.0005298511674247733,
]

M2_IS_sd = [
    0.0005380804971845754,
    0.0005219438263649965,
    0.0005262028111370937,
    0.0005601420889339825,
    0.0006013504730402632,
    0.0005732219699319367,
    0.0005386987409703083,
]

LSM_OOS_sd = [
    0.005411880208248293,
    0.0017364233562528885,
    0.0006936025757418292,
    0.0005571386786997098,
    0.00037540319500077433,
    0.00022509512520416716,
    0.0001706894927971441,
]

M1_OOS_sd = [
    0.005664506021364217,
    0.0018373008388416102,
    0.0006928709604212039,
    0.0005611459017360761,
    0.00040036150020025287,
    0.00023438472612967437,
    0.0001818519547620555,
]

M2_OOS_sd = [
    0.005599315835939781,
    0.001817489034393121,
    0.000700942731541694,
    0.000539778074019719,
    0.0003877145207962494,
    0.00022190319966563195,
    0.00017265014066886715,
]


N = [100, 1000, 5000, 10000, 20000, 50000, 100000]
N_labels = [str(n) for n in N]

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

# Convert N to string for categorical x-axis
N_labels = [str(n) for n in N]

# Define the width for the offset within each category
offset = 0.2  # Adjust this to control the spacing between the results

# Create positions for each group within the same category
N_numeric = np.arange(
    len(N)
)  # These are the positions for the categories on the x-axis

# Plot LSM_IS_result with a slight offset#343a40
ax.errorbar(
    N_numeric - offset,
    LSM_OOS_result,
    yerr=LSM_OOS_sd,
    fmt="o",
    color=colors["dark_blue"],
    label="LSM",
    markersize=8,
    capsize=5,
    elinewidth=1,
    ecolor="black",
)

# Plot M1_IS_result with a slight offset
ax.errorbar(
    N_numeric,
    M1_OOS_result,
    yerr=M1_OOS_sd,
    fmt="o",
    color=colors["light_petrol"],
    label="Delta M1",
    markersize=8,
    capsize=5,
    elinewidth=1,
    ecolor="black",
)

# Plot M2_IS_result with a slight offset
ax.errorbar(
    N_numeric + offset,
    M2_OOS_result,
    yerr=M2_OOS_sd,
    fmt="o",
    color=colors["medium_petrol"],
    label="Delta M2",
    markersize=8,
    capsize=5,
    elinewidth=1,
    ecolor="black",
)

# Customize x-ticks to display categorical values (N)
ax.set_xticks(N_numeric)
ax.set_xticklabels(N_labels)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add labels and title
plt.xlabel("Number of Out-of-sample simulations")
plt.ylabel("Price estimates")

# Add a grid for readability
plt.grid(axis="y", alpha=0.7)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="LSM",
        markerfacecolor=colors["dark_blue"],
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Delta M1",
        markerfacecolor=colors["light_petrol"],
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Delta M2",
        markerfacecolor=colors["medium_petrol"],
        markersize=8,
    ),
]

# Add the custom legend
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False,
)

# Adjust layout and show
plt.tight_layout()
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\OOS_OOS_test_vasicek.png",
    bbox_inches="tight",
)
plt.show()
