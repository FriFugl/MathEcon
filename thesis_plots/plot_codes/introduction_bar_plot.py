import numpy as np
import matplotlib.pyplot as plt

from thesis_plots.plot_codes._parameter_config import colors

dates = ["Jun-2020", "Jun-2021", "Jun-2022", "Jun-2023", "Jun-2024"]

otc_interest_derivatives = [495.141, 488.099, 502.462, 573.587, 578.805]
otc_fx_derivatives = [93.811, 102.471, 109.585, 118.467, 129.885]
otc_equity_derivatives = [6.457, 7.506, 6.989, 7.837, 8.686]
otc_commodity_contracts = [2.099, 2.453, 2.979, 2.277, 2.748]
otc_credit_derivatives = [9.05, 9.121, 9.542, 10.12, 9.196]
otc_other = [0.262, 0.347, 0.574, 0.592, 0.518]


# bars = np.add(bars1, bars2).tolist()
r = [0, 1, 2, 3, 4]
barWidth = 0.5

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))


plt.bar(
    r,
    otc_interest_derivatives,
    color=colors["dark_blue"],
    edgecolor="white",
    width=barWidth,
    label="Interest rate derivatives",
)

plt.bar(
    r,
    otc_fx_derivatives,
    bottom=otc_interest_derivatives,
    color=colors["dark_green"],
    edgecolor="white",
    width=barWidth,
    label="Foreign exchange",
)
height = np.add(otc_interest_derivatives, otc_fx_derivatives).tolist()

plt.bar(
    r,
    otc_equity_derivatives,
    bottom=height,
    color=colors["dark_petrol"],
    edgecolor="white",
    width=barWidth,
    label="Equity",
)
height = np.add(height, otc_equity_derivatives).tolist()

plt.bar(
    r,
    otc_commodity_contracts,
    bottom=height,
    color=colors["dark_grey"],
    edgecolor="white",
    width=barWidth,
    label="Commodities",
)
height = np.add(height, otc_commodity_contracts).tolist()

plt.bar(
    r,
    otc_credit_derivatives,
    bottom=height,
    color=colors["yellow"],
    edgecolor="white",
    width=barWidth,
    label="Credit",
)
height = np.add(height, otc_credit_derivatives).tolist()

plt.bar(
    r,
    otc_other,
    bottom=height,
    color=colors["medium_red"],
    edgecolor="white",
    width=barWidth,
    label="Other derivatives",
)

for i in range(len(r)):
    ax.text(
        r[i] - 0.1,
        otc_interest_derivatives[i] / 2,
        f"{otc_interest_derivatives[i]:.1f}",
        fontsize=9,
        color="white",
    )
    ax.text(
        r[i] - 0.1,
        otc_interest_derivatives[i] + otc_fx_derivatives[i] / 2,
        f"{otc_fx_derivatives[i]:.1f}",
        fontsize=9,
        color="white",
    )
    ax.text(
        r[i] - 0.1,
        otc_interest_derivatives[i]
        + otc_fx_derivatives[i]
        + otc_equity_derivatives[i]
        + otc_commodity_contracts[i]
        + otc_credit_derivatives[i]
        + otc_other[i]
        + 5,
        f"{otc_interest_derivatives[i] + otc_fx_derivatives[i] + otc_equity_derivatives[i] + otc_commodity_contracts[i] + otc_credit_derivatives[i] + otc_other[i]:.1f}",
        fontsize=9,
        color="Black",
    )


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(r, dates)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
ax.tick_params(axis="x", which="both", length=0)
plt.tight_layout()
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\intro_plot.png",
    bbox_inches="tight",
)
plt.show()
