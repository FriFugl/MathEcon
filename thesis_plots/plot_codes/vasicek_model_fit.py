from _short_rate_models import VasicekModel

from scipy.optimize import minimize

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_zcb_prices

from thesis_plots.plot_codes._parameter_config import colors

def fit_vasicek(params: tuple[float, float, float, float], P_M: dict[float, float]) -> float:
    """
    Function to fit the Vasicek model to market spot rates or ZCB prices.

    Function calculates spot rates or ZCB prices in the Vasicek model using the parameters in params and reports the sum of squared
    errors between these and the ones given in R_M. The function can ideally be used with scipy.minimize to find the
    Vasicek parameters which give the best spot rate fit to market spot rates.

    params: Tuple with (r_0, a, b, sigma)
    P_M: Market spot rates / ZCB prices.
    """
    r_0, a, b, sigma = params

    VasicekModelInstance = VasicekModel(a=a, b=b, sigma=sigma)
    short_rates_calibration = VasicekModelInstance.simulate(r_0=r_0,
                                                            T=30,
                                                            M=120,
                                                            N=1,
                                                            method='exact'
                                                            )
    fitted_values = VasicekModelInstance.price_zcb(short_rates=short_rates_calibration,
                                                   t=0,
                                                   maturities=list(P_M.keys()))

    loss = 0
    for t in list(P_M.keys()):
        loss += (P_M[t] - fitted_values.iloc[0][t]) ** 2

    return loss / len(P_M)

P_M = dict(zip(maturities, market_zcb_prices))

r_0 = -0.03
a = 0.15
b = 0.01
sigma = 0.01

param_0 = r_0, a, b, sigma

bounds = [
    (-0.999, 0.999),  # -1 < r_0 < 1
    (1e-5, 10),       # a
    (1e-5, 10),     # b
    (1e-5, 1),        # 0 < sigma < 1
]

fit_result = minimize(fit_vasicek,
                      param_0,
                      method='L-BFGS-B',
                      bounds=bounds,
                      args=(P_M),
                      options={'ftol': 1e-6, 'disp': True}
                      )

print(f"Parameters from the Vasicek fit: r_0 = {fit_result.x[0]}, a = {fit_result.x[1]} and b = {fit_result.x[2]},"
      f"sigma = {fit_result.x[3]}. Objective function of the fit: {fit_result.fun}")

r_0  = fit_result.x[0]
a = fit_result.x[1]
b = fit_result.x[2]
sigma = fit_result.x[3]

VasicekModelInstance = VasicekModel(a=a, b=b, sigma=sigma)

short_rates = VasicekModelInstance.simulate(r_0=r_0, T=30, M=120, N=1, method='exact')

fitted_zcb_prices = VasicekModelInstance.price_zcb(short_rates=short_rates,t=0, maturities=list(P_M.keys()))

result = fitted_zcb_prices.iloc[0][0.25:].tolist()
fig, ax = plt.subplots(dpi=300, figsize=(8,6))

plt.plot(maturities, market_zcb_prices, label=f'Market ZCB prices', color=colors['dark_grey'], alpha=0.7)
plt.plot(maturities, result, label=f'Fitted ZCB prices', color=colors['dark_blue'], alpha=0.7)

plt.xlabel('Maturities', fontsize=10)
plt.ylabel("ZCB price", fontsize=10)
plt.legend()

plt.grid(visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.505, -0.075), ncol=3, frameon=False)


plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\vasicek_zcb_fit.png', bbox_inches='tight')
plt.show()

APE = [abs((market_zcb_prices[i] - result[i])/market_zcb_prices[i]) for i in range(len(result))]
MAPE = sum(APE) / len(APE)
print(f"Mean absolute percentage error: {MAPE}")

fig, ax = plt.subplots(dpi=300, figsize=(8,6))
plt.plot(maturities, APE, label='Relative error', linestyle='none', marker='o', color=colors['dark_blue'], alpha=0.7)
plt.axhline(MAPE, label='Mean relative error', linestyle='--', color=colors['dark_grey'], linewidth=1.5)

plt.xlabel('Maturities', fontsize=10)
plt.ylabel("Error", fontsize=10)
plt.legend()

plt.grid(visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

ax.legend(loc='upper center', bbox_to_anchor=(0.53, -0.075), ncol=3, frameon=False)
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\vasicek_relative_erros.png', bbox_inches='tight')
plt.show()
