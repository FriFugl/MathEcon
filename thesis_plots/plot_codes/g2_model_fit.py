from _short_rate_models import GaussianModel

from scipy.optimize import minimize

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import market_zcb_prices

from thesis_plots.plot_codes._parameter_config import colors


def fit_gaussian(
    params: tuple[float, float, float, float, float],
    f_M: dict[float, float],
    P_M: dict[float, float],
) -> float:
    """
    Function to fit the 2D Gayssian model to market spot rates.

    Function calculates spot rates in the 2D Gaussian model using the parameters in params and reports the sum of squared
    errors between these and the ones given in R_M. The function can ideally be used with scipy.minimize to find the
    Vasicek parameters which give the best spot rate fit to market spot rates.

    params: Tuple with (r_0, a, b, sigma, eta, rho)
    f_M; Market forward rates.
    P_M: Market spot rates / ZCB prices.
    """
    a, b, sigma, eta, rho = params

    GaussianModelInstance = GaussianModel(
        a=a, b=b, sigma=sigma, eta=eta, rho=rho, instant_forward_rates=f_M
    )

    short_rates_calibration, x_calibration, y_calibration, varphi_calibration = (
        GaussianModelInstance.simulate(T=30, M=120, N=1, method="euler")
    )

    fitted_values = GaussianModelInstance.price_zcb(
        x_paths=x_calibration,
        y_paths=y_calibration,
        varphi=varphi_calibration,
        t=0,
        maturities=list(P_M.keys()),
    )

    loss = 0
    for t in list(P_M.keys()):
        loss += (P_M[t] - fitted_values.iloc[0][t]) ** 2

    return loss / len(P_M)


P_M = dict(zip(maturities, market_zcb_prices))
f_M = dict(zip(maturities, market_forward_rates))

r_0 = -0.03
a = 0.4
b = 0.2
sigma = 0.015
eta = 0.015
rho = -0.75
theta = 0.5

param_0 = a, b, sigma, eta, rho

bounds = [
    (1e-5, 10),  # 0 < a < 10
    (1e-5, 10),  # 0 < b < 10
    (1e-5, 1),  # 0 < sigma < 1
    (1e-5, 1),  # 0 < eta < 1
    (-1, 1),  # -1 < rho < 1
]

fit_result = minimize(
    fit_gaussian,
    param_0,
    method="L-BFGS-B",
    bounds=bounds,
    args=(f_M, P_M),
    options={"ftol": 1e-6, "disp": True},
)

a = fit_result.x[0]
b = fit_result.x[1]
sigma = fit_result.x[2]
eta = fit_result.x[3]
rho = fit_result.x[4]

print(
    f"Parameters from the 2D Gaussian model fit: a = {a}, b = {b} sigma = {sigma} and eta = {eta},"
    f"rho = {rho}. Objective function of the fit: {fit_result.fun}"
)


instant_forward_rates = dict(zip(maturities, market_forward_rates))

GaussianModelInstance = GaussianModel(
    a=a, b=b, sigma=sigma, eta=eta, rho=rho, instant_forward_rates=instant_forward_rates
)
short_rates, x, y, varphi = GaussianModelInstance.simulate(
    T=30, M=120, N=1, method="euler"
)
fitted_zcb_prices = GaussianModelInstance.price_zcb(
    x_paths=x, y_paths=y, varphi=varphi, t=0, maturities=list(P_M.keys())
)

result = fitted_zcb_prices.iloc[0].tolist()

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))

plt.plot(
    maturities,
    market_zcb_prices,
    label=f"Market ZCB prices",
    color=colors["dark_grey"],
    alpha=0.7,
)
plt.plot(
    maturities,
    result,
    label=f"Fitted ZCB prices",
    color=colors["dark_green"],
    alpha=0.7,
)

plt.xlabel("Maturities", fontsize=10)
plt.ylabel("ZCB price", fontsize=10)

plt.grid(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.505, -0.075), ncol=3, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\gaussian_zcb_fit.png",
    bbox_inches="tight",
)
plt.show()

APE = [
    abs((market_zcb_prices[i] - result[i]) / market_zcb_prices[i])
    for i in range(len(result))
]
MAPE = sum(APE) / len(APE)
print(f"Mean relative error: {MAPE}")

fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
plt.plot(
    maturities,
    APE,
    label="Relative error",
    linestyle="none",
    marker="o",
    color=colors["dark_green"],
    alpha=0.7,
)
plt.axhline(
    MAPE,
    label="Mean relative error",
    linestyle="--",
    color=colors["dark_grey"],
    linewidth=1.5,
)

plt.xlabel("Maturities", fontsize=10)
plt.ylabel("Relative error", fontsize=10)

plt.grid(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.legend(loc="upper center", bbox_to_anchor=(0.53, -0.075), ncol=3, frameon=False)
plt.savefig(
    r"G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\gaussian_relative_erros.png",
    bbox_inches="tight",
)
plt.show()
