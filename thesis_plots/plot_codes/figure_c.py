import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.plot_codes._parameter_config import colors

PP_LSM_IS = [
    4.476,
    4.835,
    7.100,
    8.504,
    3.249,
    3.739,
    6.147,
    7.665,
    2.312,
    2.881,
    5.311,
    6.917,
    1.616,
    2.211,
    4.583,
    6.242,
    1.110,
    1.689,
    3.947,
    5.638,
]

PP_LSM_IS_se = [
    0.005,
    0.006,
    0.010,
    0.013,
    0.005,
    0.006,
    0.010,
    0.014,
    0.005,
    0.006,
    0.009,
    0.012,
    0.004,
    0.005,
    0.009,
    0.013,
    0.004,
    0.006,
    0.010,
    0.013,
]

PP_delta_LSM_IS = [
    4.477,
    4.838,
    7.101,
    8.504,
    3.250,
    3.741,
    6.146,
    7.665,
    2.313,
    2.883,
    5.311,
    6.917,
    1.616,
    2.212,
    4.582,
    6.241,
    1.110,
    1.690,
    3.946,
    5.638,
]

PP_delta_LSM_IS_se = [
    0.005,
    0.006,
    0.010,
    0.013,
    0.005,
    0.006,
    0.010,
    0.014,
    0.005,
    0.006,
    0.010,
    0.011,
    0.004,
    0.005,
    0.009,
    0.014,
    0.004,
    0.006,
    0.010,
    0.013,
]

PP_LSM_OOS = [
    4.476,
    4.835,
    7.098,
    8.501,
    3.248,
    3.740,
    6.143,
    7.664,
    2.313,
    2.881,
    5.308,
    6.911,
    1.615,
    2.209,
    4.580,
    6.240,
    1.108,
    1.687,
    3.945,
    5.638,
]

PP_LSM_OOS_se = [
    0.005,
    0.006,
    0.010,
    0.013,
    0.005,
    0.006,
    0.010,
    0.014,
    0.005,
    0.006,
    0.010,
    0.011,
    0.004,
    0.005,
    0.009,
    0.014,
    0.004,
    0.006,
    0.010,
    0.013,
]

PP_delta_LSM_OOS = [
    4.477,
    4.839,
    7.100,
    8.504,
    3.250,
    3.744,
    6.146,
    7.667,
    2.314,
    2.884,
    5.310,
    6.914,
    1.616,
    2.211,
    4.581,
    6.242,
    1.109,
    1.690,
    3.947,
    5.642,
]

PP_delta_LSM_OOS_se = [
    0.005,
    0.007,
    0.011,
    0.015,
    0.006,
    0.006,
    0.008,
    0.014,
    0.005,
    0.006,
    0.010,
    0.012,
    0.004,
    0.005,
    0.009,
    0.015,
    0.004,
    0.005,
    0.009,
    0.013,
]

LP_LSM_IS = [
    4.467,
    4.828,
    7.114,
    8.520,
    3.238,
    3.733,
    6.157,
    7.682,
    2.303,
    2.879,
    5.313,
    6.929,
    1.616,
    2.211,
    4.593,
    6.263,
    1.110,
    1.693,
    3.949,
    5.648,
]

LP_LSM_IS_se = [
    0.034,
    0.035,
    0.066,
    0.075,
    0.029,
    0.034,
    0.065,
    0.074,
    0.026,
    0.035,
    0.057,
    0.061,
    0.022,
    0.030,
    0.056,
    0.067,
    0.019,
    0.028,
    0.053,
    0.061,
]

LP_delta_LSM_IS = [
    4.475,
    4.843,
    7.109,
    8.509,
    3.252,
    3.750,
    6.152,
    7.672,
    2.311,
    2.885,
    5.306,
    6.919,
    1.621,
    2.213,
    4.586,
    6.253,
    1.111,
    1.692,
    3.943,
    5.637,
]

LP_delta_LSM_IS_se = [
    0.031,
    0.035,
    0.061,
    0.075,
    0.028,
    0.035,
    0.063,
    0.077,
    0.026,
    0.034,
    0.058,
    0.059,
    0.023,
    0.029,
    0.055,
    0.067,
    0.018,
    0.027,
    0.051,
    0.060,
]

LP_LSM_OOS = [
    4.456,
    4.827,
    7.101,
    8.494,
    3.244,
    3.724,
    6.149,
    7.637,
    2.282,
    2.859,
    5.294,
    6.871,
    1.605,
    2.204,
    4.576,
    6.230,
    1.117,
    1.685,
    3.942,
    5.641,
]

LP_LSM_OOS_se = [
    0.085,
    0.098,
    0.188,
    0.249,
    0.096,
    0.098,
    0.182,
    0.216,
    0.083,
    0.106,
    0.185,
    0.192,
    0.076,
    0.104,
    0.168,
    0.229,
    0.065,
    0.086,
    0.155,
    0.188,
]

LP_delta_LSM_OOS = [
    4.469,
    4.848,
    7.104,
    8.508,
    3.262,
    3.745,
    6.162,
    7.647,
    2.290,
    2.872,
    5.300,
    6.884,
    1.614,
    2.211,
    4.584,
    6.240,
    1.123,
    1.693,
    3.951,
    5.653,
]

LP_delta_LSM_OOS_se = [
    0.085,
    0.107,
    0.186,
    0.252,
    0.103,
    0.101,
    0.182,
    0.224,
    0.084,
    0.105,
    0.187,
    0.196,
    0.074,
    0.104,
    0.163,
    0.228,
    0.062,
    0.088,
    0.154,
    0.192,
]

scenario = [i for i in range(1, 21)]


LSM_IS_diff = [PP - LP for PP, LP in zip(PP_LSM_IS, LP_LSM_IS)]
delta_LSM_IS_diff = [PP - LP for PP, LP in zip(PP_delta_LSM_IS, LP_delta_LSM_IS)]

LSM_IS_se_ratio = [LP / PP for PP, LP in zip(PP_LSM_IS_se, LP_LSM_IS_se)]
delta_LSM_IS_se_ratio = [
    LP / PP for PP, LP in zip(PP_delta_LSM_IS_se, LP_delta_LSM_IS_se)
]

LSM_OOS_diff = [PP - LP for PP, LP in zip(PP_LSM_OOS, LP_LSM_OOS)]
delta_LSM_OOS_diff = [PP - LP for PP, LP in zip(PP_delta_LSM_OOS, LP_delta_LSM_OOS)]

LSM_OOS_se_ratio = [LP / PP for PP, LP in zip(PP_LSM_OOS_se, LP_LSM_OOS_se)]
delta_LSM_OOS_se_ratio = [
    LP / PP for PP, LP in zip(PP_delta_LSM_OOS_se, LP_delta_LSM_OOS_se)
]

# IS difference plot
# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    scenario,
    LSM_IS_diff,
    linestyle="none",
    color=colors["dark_red"],
    marker="o",
    label="LSM",
)
plt.axhline(
    np.mean(LSM_IS_diff), linestyle="--", color=colors["dark_red"], linewidth=1.5
)

plt.plot(
    scenario,
    delta_LSM_IS_diff,
    linestyle="none",
    color=colors["dark_blue"],
    marker="o",
    label="Delta LSM",
)
plt.axhline(
    np.mean(delta_LSM_IS_diff), linestyle="--", color=colors["dark_blue"], linewidth=1.5
)

ax.tick_params(axis="x", which="both", bottom=False, top=False)
plt.xticks([1, 5, 10, 15, 20])
plt.grid(axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\implementation_test_IS.png",
    bbox_inches="tight",
)
plt.show()

# OOS difference plot
# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    scenario,
    LSM_OOS_diff,
    linestyle="none",
    color=colors["dark_red"],
    marker="o",
    label="LSM",
)
plt.axhline(
    np.mean(LSM_OOS_diff), linestyle="--", color=colors["dark_red"], linewidth=1.5
)

plt.plot(
    scenario,
    delta_LSM_OOS_diff,
    linestyle="none",
    color=colors["dark_blue"],
    marker="o",
    label="Delta LSM",
)
plt.axhline(
    np.mean(delta_LSM_OOS_diff),
    linestyle="--",
    color=colors["dark_blue"],
    linewidth=1.5,
)

ax.tick_params(axis="x", which="both", bottom=False, top=False)
plt.xticks([1, 5, 10, 15, 20])
plt.grid(axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\implementation_test_OOS.png",
    bbox_inches="tight",
)
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    scenario,
    LSM_IS_se_ratio,
    linestyle="none",
    color=colors["dark_red"],
    marker="o",
    label="LSM",
)
plt.axhline(
    np.mean(LSM_IS_se_ratio), linestyle="--", color=colors["dark_red"], linewidth=1.5
)

plt.plot(
    scenario,
    delta_LSM_IS_se_ratio,
    linestyle="none",
    color=colors["dark_blue"],
    marker="o",
    label="Delta LSM",
)
plt.axhline(
    np.mean(delta_LSM_IS_se_ratio),
    linestyle="--",
    color=colors["dark_blue"],
    linewidth=1.5,
)

ax.tick_params(axis="x", which="both", bottom=False, top=False)
plt.xticks([1, 5, 10, 15, 20])
plt.grid(axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\implementation_test_IS_se.png",
    bbox_inches="tight",
)
plt.show()


fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    scenario,
    LSM_OOS_se_ratio,
    linestyle="none",
    color=colors["dark_red"],
    marker="o",
    label="LSM",
)
plt.axhline(
    np.mean(LSM_OOS_se_ratio), linestyle="--", color=colors["dark_red"], linewidth=1.5
)

plt.plot(
    scenario,
    delta_LSM_OOS_se_ratio,
    linestyle="none",
    color=colors["dark_blue"],
    marker="o",
    label="Delta LSM",
)
plt.axhline(
    np.mean(delta_LSM_OOS_se_ratio),
    linestyle="--",
    color=colors["dark_blue"],
    linewidth=1.5,
)

ax.tick_params(axis="x", which="both", bottom=False, top=False)
plt.xticks([1, 5, 10, 15, 20])
plt.grid(axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\implementation_test_OOS_se.png",
    bbox_inches="tight",
)
plt.show()
