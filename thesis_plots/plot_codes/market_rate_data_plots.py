import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

from _short_rate_models import VasicekModel
from _short_rate_models import GaussianModel

from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import market_spot_rates
from thesis_plots.plot_codes._parameter_config import vasicek_r_0
from thesis_plots.plot_codes._parameter_config import vasicek_a
from thesis_plots.plot_codes._parameter_config import vasicek_b
from thesis_plots.plot_codes._parameter_config import vasicek_sigma
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_rho
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_eta

from thesis_plots.plot_codes._parameter_config import colors

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    maturities,
    market_spot_rates,
    label="Spot rate",
    color=colors["dark_petrol"],
    alpha=0.7,
)
plt.plot(
    maturities,
    market_forward_rates,
    label="Forward rate",
    color=colors["yellow"],
    alpha=0.7,
)

plt.xlabel("Maturities", fontsize=10)
plt.legend()

ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

plt.grid(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper center", bbox_to_anchor=(0.524, -0.075), ncol=3, frameon=False)

plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\market_rate_data.png",
    bbox_inches="tight",
)
plt.show()
