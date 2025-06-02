import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_plots.plot_codes._parameter_config import colors

backwards_pass_simulation_time_T = [
    np.float64(0.018489210605621337),
    np.float64(0.05323436737060547),
    np.float64(0.10700492143630981),
]

forward_pass_simulation_time_T = [
    np.float64(0.009496266841888428),
    np.float64(0.03022165298461914),
    np.float64(0.06200347423553467),
]

LSM_calibration_times_T = [
    np.float64(0.02805861711502075),
    np.float64(0.09551172256469727),
    np.float64(0.1806342911720276),
]

M1_calibration_times_T = [
    np.float64(0.03748821020126343),
    np.float64(0.13705304622650147),
    np.float64(0.2590860390663147),
]

M2_calibration_times_T = [
    np.float64(0.031170415878295898),
    np.float64(0.09273351907730103),
    np.float64(0.1653588342666626),
]

LSM_estimation_times_T = [
    np.float64(0.006364459991455078),
    np.float64(0.015597219467163087),
    np.float64(0.025175647735595705),
]

M1_estimation_times_T = [
    np.float64(0.006108746528625488),
    np.float64(0.015344514846801757),
    np.float64(0.024644715785980223),
]

M2_estimation_times_T = [
    np.float64(0.006090483665466309),
    np.float64(0.015352423191070557),
    np.float64(0.024655725955963135),
]

data = {
    "Category": ["LSM", "LSM", "Delta M1", "Delta M1", "Delta M2"],
    "Subcategory": ["Vasicek", "G2++"] * 2 + ["Vasicek"],
    "Backwards": [
        np.float64(0.09551172256469727),
        np.float64(0.18177083253860474),
        np.float64(0.13705304622650147),
        np.float64(0.19116333484649659),
        np.float64(0.09273351907730103),
    ],
    "Forward": [
        np.float64(0.015597219467163087),
        np.float64(0.0160345721244812),
        np.float64(0.015344514846801757),
        np.float64(0.015752747058868408),
        np.float64(0.015352423191070557),
    ],
    "IS": [
        np.float64(0.05323436737060547),
        np.float64(0.08099297523498535),
        np.float64(0.05323436737060547),
        np.float64(0.08099297523498535),
        np.float64(0.05323436737060547),
    ],
    "OOS": [
        np.float64(0.03022165298461914),
        np.float64(0.04532495498657227),
        np.float64(0.03022165298461914),
        np.float64(0.04532495498657227),
        np.float64(0.03022165298461914),
    ],
}

df = pd.DataFrame(data)

# Create a combined label for X-axis
df["Label"] = df["Category"] + " - " + df["Subcategory"]

# Position for each bar
x = np.arange(len(df))

# Plotting the stacked bars
fig, ax = plt.subplots(figsize=(10, 6))


bar1 = ax.bar(
    x, df["IS"], label="Backward pass path simulation", color=colors["dark_grey"]
)
bar2 = ax.bar(
    x,
    df["OOS"],
    bottom=df["IS"],
    label="Forward pass path simulation",
    color=colors["medium_grey"],
)
bar3 = ax.bar(
    x,
    df["Backwards"],
    bottom=df["IS"] + df["OOS"],
    label="Backward pass",
    color=colors["light_grey"],
)
bar4 = ax.bar(
    x,
    df["Forward"],
    bottom=df["IS"] + df["OOS"] + df["Backwards"],
    label="Forward pass",
    color=colors["dark_red"],
)

ax.set_xticks(x)
ax.set_xticklabels(df["Label"], rotation=45)
ax.set_ylabel("Run-time (seconds)")
ax.set_yticks([0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
ax.tick_params(axis="x", which="both", bottom=False, top=False)
plt.savefig(
    rf"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\run_time_test.png",
    bbox_inches="tight",
)
plt.show()
