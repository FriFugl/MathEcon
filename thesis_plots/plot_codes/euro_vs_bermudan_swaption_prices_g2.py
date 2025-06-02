import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.special import ndtr
from scipy.integrate import quad

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel
from _LSM import LSM_method_v2

import thesis_plots.plot_codes._parameter_config as cfg


def V(t, T, a, b, sigma, eta, rho):
    return (((sigma ** 2) / (a ** 2))
            * (T- t + (2 / a) * np.exp(-a * (T - t)) - (1 / (2 * a)) * np.exp(-2 * a * (T - t)) - (3 / (2 * a)))
            + ((eta ** 2) / (b ** 2))
            * (T- t + (2 / b) * np.exp(-b * (T - t)) - (1 / (2 * b)) * np.exp(-2 * b * (T - t)) - (3 / (2 * b)))
            + ((2 * rho * sigma * eta) / (a * b))
            * (T - t + (np.exp(-a * (T - t)) - 1) / a + (np.exp(-b * (T - t)) - 1) / b - (np.exp(-(a + b) * (T - t)) - 1) / (a + b)))

def y_bar_equation(y_bar, x, strike, T, swap_annuities, a, b, sigma, eta, rho):
    _sum = 0
    for i in range(len(swap_annuities)):
        t_i = swap_annuities[i]
        B_b = ((1 - np.exp(-b * (t_i - T))) / b)
        B_a = ((1 - np.exp(-a * (t_i - T))) / a)

        A = (P_M[t_i] / P_M[T]) * np.exp(0.5 * (V(T, t_i, a, b, sigma, eta, rho) - V(0, t_i, a, b, sigma, eta, rho) + V(0, T, a, b, sigma, eta, rho)))

        if swap_annuities[i] == swap_annuities[-1]:
            c = 1 + strike * (swap_annuities[i] - swap_annuities[i-1])
        else:
            if i == 0:
                c = strike * (swap_annuities[i] - T)
            else:
                c = strike * (swap_annuities[i] - swap_annuities[i-1])


        _sum += c * A * np.exp(-B_a * x - B_b*y_bar)

    return 1 - _sum

def integrand(x, strike, omega, T, swap_annuities, P_M, a, b, sigma, eta, rho):

    sigma_x = sigma * np.sqrt((1 - np.exp(-2 * a * T))/(2 * a))
    sigma_y = eta * np.sqrt((1 - np.exp(-2 * b * T))/(2 * b))

    rho_xy = ((rho * sigma * eta) / ((a + b) * sigma_x * sigma_y)) * (1 - np.exp(-(a + b) * T))

    M_x_T = (((sigma**2)/(a**2) + (rho * sigma * eta) / (a * b)) * (1 - np.exp(-a * T))
             - ((sigma**2) / (2 * a**2)) * (1 - np.exp(-2 * a * T)) - ((rho * sigma * eta) / (b * (a + b))) * (1 - np.exp(-(a + b) * T)))

    M_y_T = (((eta**2)/(b**2) + (rho * sigma * eta) / (a * b)) * (1 - np.exp(-b * T))
             - ((eta**2) / (2 * b**2)) * (1 - np.exp(-2 * b * T)) - ((rho * sigma * eta) / (a * (a + b))) * (1 - np.exp(-(a + b) * T)))

    mu_x = -M_x_T
    mu_y = -M_y_T

    y_bar = fsolve(y_bar_equation, x0=0.05, args=(x, strike, T, swap_annuities, a, b, sigma, eta, rho))[0]

    h_1 = ((y_bar - mu_y) / (sigma_y * np.sqrt(1 - rho_xy ** 2))) - (rho_xy * (x - mu_x)) / (
                sigma_x * np.sqrt(1 - rho_xy ** 2))

    _sum = 0
    for i in range(len(swap_annuities)):
        t_i = swap_annuities[i]
        B_b = ((1 - np.exp(-b * (t_i - T))) / b)
        B_a = ((1 - np.exp(-a * (t_i - T))) / a)

        A = (P_M[t_i] / P_M[T]) * np.exp(0.5 * (V(T, t_i, a, b, sigma, eta, rho) - V(0, t_i, a, b, sigma, eta, rho) + V(0, T, a, b, sigma, eta, rho)))

        h_2 = h_1 + B_b * sigma_y * np.sqrt(1 - rho_xy**2)

        if swap_annuities[i] == swap_annuities[-1]:
            c = 1 + strike * (swap_annuities[i] - swap_annuities[i-1])
        else:
            if i == 0:
                c = strike * (swap_annuities[i] - T)
            else:
                c = strike * (swap_annuities[i] - swap_annuities[i-1])

        _lambda = c * A * np.exp(-B_a * x)
        kappa = -B_b * (mu_y - 0.5 * (1 - rho_xy**2) * (sigma_y**2) * B_b + rho_xy * sigma_y * ((x - mu_x) / sigma_x))

        _sum += _lambda * np.exp(kappa) * ndtr(-omega * h_2)

    return ((np.exp(-0.5 * (((x - mu_x) / sigma_x)**2))) / (sigma_x * np.sqrt(2 * np.pi))) * (ndtr(-omega * h_1) - _sum)

a = cfg.g2_a
b = cfg.g2_b

sigma = cfg.g2_sigma
eta = cfg.g2_eta
rho = cfg.g2_rho

maturities = cfg.maturities
market_spot_rates = cfg.market_spot_rates
market_zcb_prices = cfg.market_zcb_prices
market_forward_rates = cfg.market_forward_rates

P_M = dict(zip(maturities, market_zcb_prices))

final_expiry = 10

M = cfg.M
alpha = cfg.alpha
strike = cfg.strike

omega = 1
expiry_dates = [i for i in range(1, final_expiry)]

euro_prices = []
for T in expiry_dates:
    swap_annuities = [i for i in range(T + 1, final_expiry + 1)]
    result, err = quad(integrand, -10, 10, args=(strike, omega, T, swap_annuities, P_M, a, b, sigma, eta, rho))

    euro_prices.append(result)
    print(f"Euro {T}Y{len(swap_annuities)}Y Swaption price = {result} ({err})")

instant_forward_rates = dict(zip(maturities, market_forward_rates))

GaussianModelInstance =  GaussianModel(a=a, b=b, sigma=sigma, eta=eta, rho=rho,
                                          instant_forward_rates=instant_forward_rates)

exercise_dates = [i * (final_expiry / M) for i in range(1, M + 1) if i * (final_expiry / M) < final_expiry - alpha]
LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=final_expiry,
                                                                                                           M=M,
                                                                                                           N=10000,
                                                                                                           method='euler')

short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=final_expiry,
                                                                                                           M=M,
                                                                                                           N=10000,
                                                                                                           method='euler'
                                                                                                       )

swap_rates_calibration, accrual_factors_calibration = GaussianModelInstance.swap_rate(x_paths=x_calibration,
                                                                                      y_paths=y_calibration,
                                                                                      varphi=varphi_calibration,
                                                                                      entry_dates=exercise_dates,
                                                                                      expiry=final_expiry,
                                                                                      alpha=alpha)

swap_rates_estimation, accrual_factors_estimation = GaussianModelInstance.swap_rate(x_paths=x_estimation,
                                                                                    y_paths=y_estimation,
                                                                                    varphi=varphi_estimation,
                                                                                    entry_dates=exercise_dates,
                                                                                    expiry=final_expiry,
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

LSM_OOS = LSM.estimation(underlying_asset_paths=swap_rates_estimation.copy(),
                         payoffs=estimation_payoffs.copy(),
                         discount_factors=discount_factors_estimation.copy(),
                         betas=LSM_betas.copy())

print(f"10Y no-call 1 Bermudan price, LSM = {LSM_IS, LSM_OOS}")

prices = [LSM_OOS] + euro_prices
bars = ['LSM'] + [f'{i}Y{final_expiry-i}Y' for i in range(1, final_expiry)]

colors =  [cfg.colors['dark_green']] + [cfg.colors['light_green'] for i in range(final_expiry)]
# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))
plt.bar(bars, prices, color=colors, edgecolor='White')

ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])

# Add labels and title
plt.ylabel('Swaption prices')
plt.xticks(rotation=45)

# Show grid for better readability
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Display the plot
plt.tight_layout()
plt.savefig(fr'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\euro_vs_bermudan_swaption_prices_g2_{final_expiry}Y.png',
          bbox_inches='tight')
plt.show()