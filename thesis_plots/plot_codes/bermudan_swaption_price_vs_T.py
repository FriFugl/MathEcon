import numpy as np
import matplotlib.pyplot as plt

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel
from _short_rate_models import GaussianModel
from _LSM import LSM_method_v2

from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import vasicek_r_0
from thesis_plots.plot_codes._parameter_config import vasicek_a
from thesis_plots.plot_codes._parameter_config import vasicek_b
from thesis_plots.plot_codes._parameter_config import vasicek_sigma
from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_eta
from thesis_plots.plot_codes._parameter_config import g2_rho

from thesis_plots.plot_codes._parameter_config import colors

VasicekModelInstance = VasicekModel(a=vasicek_a, b=vasicek_b, sigma=vasicek_sigma)

instant_forward_rates = dict(zip(maturities, market_forward_rates))
GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=g2_rho,
                                      instant_forward_rates=instant_forward_rates)

vasicek_result = []
g2_result = []
expiries = [i for i in range(5, 16)]
for final_expiry in expiries:
    M = final_expiry * 1

    exercise_dates = [i * (final_expiry / M) for i in range(1, M + 1) if i * (final_expiry / M) < final_expiry - alpha]
    LSM = LSM_method_v2(strike=strike, exercise_dates=exercise_dates, degree=3)

    short_rates_calibration = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=final_expiry, M=M, N=10000,
                                                            method='exact', seed=1)
    short_rates_estimation = VasicekModelInstance.simulate(r_0=vasicek_r_0, T=final_expiry, M=M, N=10000,
                                                           method='exact', seed=2)

    swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(
        short_rate=short_rates_calibration,
        entry_dates=exercise_dates,
        expiry=final_expiry,
        alpha=alpha)

    swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(
        short_rate=short_rates_estimation,
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

    vasicek_result.append(LSM_OOS)

    short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(
        T=final_expiry,
        M=M,
        N=10000,
        method='euler',
        seed=1)
    short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(
        T=final_expiry,
        M=M,
        N=10000,
        method='euler',
        seed=2)

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

    g2_result.append(LSM_OOS)

# Plotting
fig, ax = plt.subplots(dpi=300, figsize=(8,6))

plt.plot(expiries, vasicek_result, color=colors['dark_blue'], label='Vasicek')
plt.plot(expiries, g2_result, color=colors['dark_green'], linestyle='--', label='G2++')

ax.set_xticks([i for i in range(5, 16)])

ax.set_ylim(0, 0.2)
ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2])

# Add labels and title
plt.ylabel('Swaption prices')
plt.xlabel('Maturity')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False)
# Show grid for better readability
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.5)

# Display the plot
plt.tight_layout()
plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\bermudan_swaption_price_vs_T.png',
         bbox_inches='tight')
plt.show()