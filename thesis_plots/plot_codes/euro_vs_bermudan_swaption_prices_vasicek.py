import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.special import ndtr

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel
from _LSM import LSM_method_v2

import thesis_plots.plot_codes._parameter_config as cfg

def euro_option_price_vasicek(K,T1,T2,p_T1,p_T2,a,sigma,type = "call"):
    sigma_p = (sigma/a)*(1-np.exp(-a*(T2-T1)))*np.sqrt((1-np.exp(-2*a*T1))/(2*a))
    d1 = (np.log(p_T2/(p_T1*K)))/sigma_p + 0.5*sigma_p
    d2 = d1 - sigma_p
    if type == "call":
        price = p_T2*ndtr(d1) - p_T1*K*ndtr(d2)
    elif type == "put":
        price = p_T1*K*ndtr(-d2) - p_T2*ndtr(-d1)
    return price

def vasicek_zcb_price(t, T, r_0, a, b, sigma):
    B = (1 / a) * (1 - np.exp(-a * (T - t)))
    A = ((B - T + t) * (a * b - 0.5 * (sigma ** 2))) / (
            a ** 2
    ) - ((sigma ** 2) * (B ** 2)) / (4 * a)

    return np.exp(A - r_0*B)

def r_star_equation(r_star, T, t_i, a, b, sigma, X, tau):
    sum = 0
    for time in t_i:
        B = (1 / a) * (1 - np.exp(-a * (time - T)))
        A = ((B - time + T) * (a * b - 0.5 * (sigma ** 2))) / (a ** 2) - ((sigma ** 2) * (B ** 2)) / (4 * a)

        if time == t_i[-1]:
            sum += (1 + X * tau) * np.exp(A - r_star * B)
        else:
            sum += X*tau * np.exp(A - r_star * B)

    return 1 - sum

final_expiry = 10

r_0 = cfg.vasicek_r_0
a = cfg.vasicek_a
b = cfg.vasicek_b
sigma = cfg.vasicek_sigma

M = cfg.M
alpha = cfg.alpha

X = cfg.strike
tau = alpha

expiry_dates = [i for i in range(1, final_expiry)]

euro_prices = []
for T in expiry_dates:
    t_i = [i for i in range(T+1, final_expiry+1)]


    r_star = fsolve(r_star_equation, x0=0.05, args=(T, t_i, a, b, sigma, X, tau))[0]

    sum = 0
    for time in t_i:
        X_i = vasicek_zcb_price(t=T, T=time, r_0=r_star, a=a, b=b, sigma=sigma)

        p1 = vasicek_zcb_price(t=0, T=T, r_0=r_0, a=a, b=b, sigma=sigma)
        p2 = vasicek_zcb_price(t=0, T=time, r_0=r_0, a=a, b=b, sigma=sigma)

        ZBP = euro_option_price_vasicek(K=X_i, T1=T, T2=time, p_T1=p1, p_T2=p2, a=a, sigma=sigma, type = 'put')

        if time == t_i[-1]:
            sum += (1+X*tau)*ZBP
        else:
            sum += X*tau*ZBP

    euro_prices.append(sum)
    print(f"Euro {T}Y{len(t_i)}Y Swaption price = {sum}")

VasicekModelInstance = VasicekModel(a=a, b=b, sigma=sigma)

exercise_dates = [i * (final_expiry / M) for i in range(1, M + 1) if i * (final_expiry / M) < final_expiry]
LSM = LSM_method_v2(strike=X, exercise_dates=exercise_dates, degree=3)

short_rates_calibration = VasicekModelInstance.simulate(r_0=r_0, T=final_expiry, M=M, N=10000, method='exact')
short_rates_estimation = VasicekModelInstance.simulate(r_0=r_0, T=final_expiry, M=M, N=10000, method='exact')

swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(short_rate=short_rates_calibration,
                                                                                     entry_dates=exercise_dates,
                                                                                     expiry=final_expiry,
                                                                                     alpha=alpha)

swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(short_rate=short_rates_estimation,
                                                                                   entry_dates=exercise_dates,
                                                                                   expiry=final_expiry,
                                                                                   alpha=alpha)

calibration_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_calibration,
                                                  accrual_factors=accrual_factors_calibration, strike=X)
estimation_payoffs = _calculate_swaption_payoffs(swap_rates=swap_rates_estimation,
                                                 accrual_factors=accrual_factors_estimation, strike=X)

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

colors =  [cfg.colors['dark_blue']] + [cfg.colors['light_petrol'] for i in range(final_expiry)]
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
plt.savefig(fr'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\euro_vs_bermudan_swaption_prices_vasicek_{final_expiry}Y.png',
          bbox_inches='tight')
plt.show()