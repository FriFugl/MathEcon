import matplotlib.pyplot as plt

from _helpers import _calculate_swaption_payoffs
from _helpers import _short_rate_to_discount_factors
from _short_rate_models import GaussianModel
from _LSM import LSM_method_v2

import thesis_plots.plot_codes._parameter_config as cfg

alpha = cfg.alpha
X = cfg.strike

g2_a = cfg.g2_a
g2_b = cfg.g2_b

g2_sigma = cfg.g2_sigma
eta = cfg.g2_eta
rho = cfg.g2_eta

instant_forward_rates = dict(zip(cfg.maturities, cfg.market_forward_rates))

g2_result = []

final_expiry = cfg.T
M = cfg.M

offset = [i for i in range(-100, 101, 10)]
g2_etas = [eta + (i/10000) for i in offset]
for eta in g2_etas:
    GaussianModelInstance = GaussianModel(a=g2_a, b=g2_b, sigma=g2_sigma, eta=eta, rho=rho,
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

plt.plot(offset,g2_result, label=f'LSM', color=cfg.colors['dark_green'], alpha=0.7)

plt.xlabel('Offset (bps)')
plt.ylabel("Swaption price")

plt.grid(visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(r'G:\Mit drev\Matematik-økonomi\Kandidat\Speciale\GitHub\MathEcon\thesis_plots\plots\bermudan_swaption_price_vs_eta_g2.png',
          bbox_inches='tight')

plt.show()