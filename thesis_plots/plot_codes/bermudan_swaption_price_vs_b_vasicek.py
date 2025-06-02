import numpy as np
import matplotlib.pyplot as plt

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import VasicekModel
from _LSM import LSM_method_v2

import thesis_plots.plot_codes._parameter_config as cfg

finaly_expiry = cfg.T
M = cfg.M
alpha = cfg.alpha

r_0 = cfg.vasicek_r_0
a = cfg.vasicek_a
sigma = cfg.vasicek_sigma

expiry_dates = [i for i in range(1, finaly_expiry)]

X = cfg.strike

LSM_result = []

offset = [i for i in range(-200, 201, 10)]
b_params = [cfg.vasicek_b + (i/10000) for i in offset]
for b in b_params:
    VasicekModelInstance = VasicekModel(a=a, b=b, sigma=sigma)
    exercise_dates = [i * (finaly_expiry / M) for i in range(1, M + 1) if i * (finaly_expiry / M) < finaly_expiry]
    LSM = LSM_method_v2(strike=X, exercise_dates=exercise_dates, degree=3)

    short_rates_calibration = VasicekModelInstance.simulate(r_0=r_0, T=finaly_expiry, M=M, N=10000, method='exact', seed=1)
    short_rates_estimation = VasicekModelInstance.simulate(r_0=r_0, T=finaly_expiry, M=M, N=10000, method='exact', seed=1)

    swap_rates_calibration, accrual_factors_calibration = VasicekModelInstance.swap_rate(short_rate=short_rates_calibration,
                                                                                         entry_dates=exercise_dates,
                                                                                         expiry=finaly_expiry,
                                                                                         alpha=alpha)

    swap_rates_estimation, accrual_factors_estimation = VasicekModelInstance.swap_rate(short_rate=short_rates_estimation,
                                                                                       entry_dates=exercise_dates,
                                                                                       expiry=finaly_expiry,
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

    LSM_result.append(LSM_OOS)

fig, ax = plt.subplots(dpi=300, figsize=(8,6))

plt.plot(offset,LSM_result, label=f'LSM', color=cfg.colors['dark_blue'], alpha=0.7)

plt.xlabel('Offset (bps)')
plt.ylabel("Swaption price")

plt.grid(visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\bermudan_swaption_price_vs_b_vasicek.png',
          bbox_inches='tight')

plt.show()