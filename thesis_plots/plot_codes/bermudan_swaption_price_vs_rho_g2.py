import numpy as np
import matplotlib.pyplot as plt

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel
from _LSM import LSM_method_v2


from thesis_plots.plot_codes._parameter_config import alpha
from thesis_plots.plot_codes._parameter_config import T
from thesis_plots.plot_codes._parameter_config import M
from thesis_plots.plot_codes._parameter_config import strike
from thesis_plots.plot_codes._parameter_config import maturities
from thesis_plots.plot_codes._parameter_config import market_forward_rates
from thesis_plots.plot_codes._parameter_config import g2_a
from thesis_plots.plot_codes._parameter_config import g2_b
from thesis_plots.plot_codes._parameter_config import g2_sigma
from thesis_plots.plot_codes._parameter_config import g2_eta

from thesis_plots.plot_codes._parameter_config import colors

final_expiry = T
X = strike
g2_result = []

instant_forward_rates = dict(zip(maturities, market_forward_rates))

g2_rho = [i/100 for i in range(-100, 101)]
for rho in g2_rho:
    GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=g2_eta, rho=rho,
                                          instant_forward_rates=instant_forward_rates)

    exercise_dates = [i * (final_expiry / M) for i in range(1, M + 1) if i * (final_expiry / M) < final_expiry - alpha]
    LSM = LSM_method_v2(strike=X, exercise_dates=exercise_dates, degree=3)


    short_rates_calibration, x_calibration, y_calibration, varphi_calibration = GaussianModelInstance.simulate(T=final_expiry,
                                                                                                               M=M,
                                                                                                               N=10000,
                                                                                                               method='euler',
                                                                                                               seed=1)
    short_rates_estimation, x_estimation, y_estimation, varphi_estimation = GaussianModelInstance.simulate(T=final_expiry,
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

    g2_result.append(LSM_OOS)

fig, ax = plt.subplots(dpi=300, figsize=(8,6))

plt.plot(g2_rho,g2_result, label=f'LSM', color=colors['dark_green'], alpha=0.7)

plt.xlabel(r'$\rho$')
plt.ylabel("Swaption price")

plt.grid(visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\bermudan_swaption_price_vs_rho_g2.png',
          bbox_inches='tight')


plt.show()