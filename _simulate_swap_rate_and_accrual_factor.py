import pandas as pd

from _vasicek_model import _short_rate_to_zcb_prices_vasicek
from _calculate_swap_and_swaption_details import _compute_swap_rate_and_accrual_factor


def _short_rate_to_swap_rate_and_accrual_factor_vasicek(
    short_rate: pd.DataFrame,
    expiry: float,
    exercise_dates: list,
    a: float,
    b: float,
    sigma: float,
) -> pd.DataFrame:
    swap_rates = {}
    accrual_factors = {}

    for exercise_date in exercise_dates[:-1]:
        start_date = exercise_date + 1

        swap_maturities = [i for i in range(start_date, expiry + 1)]

        zcb_prices = _short_rate_to_zcb_prices_vasicek(
            short_rates=short_rate,
            t=exercise_date,
            maturities=swap_maturities,
            a=a,
            b=b,
            sigma=sigma,
        )

        R, S = _compute_swap_rate_and_accrual_factor(
            zcb_prices=zcb_prices, start=start_date, maturity=expiry
        )

        swap_rates[exercise_date] = R
        accrual_factors[exercise_date] = S

    swap_rates = pd.DataFrame(swap_rates)
    accrual_factors = pd.DataFrame(accrual_factors)

    return swap_rates, accrual_factors
