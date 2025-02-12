import pandas as pd
import numpy as np
from _calculate_swap_and_swaption_details import _calculate_swaption_payoffs


def _LSM_method(
    short_rates: pd.DataFrame,
    swap_rates: pd.DataFrame,
    accrual_factors: pd.DataFrame,
    strike: float,
    N: int,
) -> pd.DataFrame:

    payoffs = _calculate_swaption_payoffs(
        swap_rates=swap_rates, accrual_factors=accrual_factors, strike=strike
    )

    for i, col in enumerate(reversed(payoffs.columns)):
        if i == 0:
            cashflows = payoffs[col]
            continue

        discount = np.exp(
            -short_rates[col].iloc[0]*(short_rates[col+1]-short_rates[col])
        )  # NEEDS TO BE FIXED FOR TIME INTERVAL FOR DISCOUNTING

        itm_paths = payoffs.index[payoffs[col] > 0].tolist()
        if itm_paths == []:
            cashflows = cashflows * discount
            continue

        itm_swap_rates = swap_rates.loc[itm_paths, col]
        itm_cashflows = cashflows[itm_paths].to_numpy()

        X = np.column_stack(
            (np.ones(len(itm_swap_rates)), itm_swap_rates, itm_swap_rates**2)
        )
        beta, residuals, rank, s = np.linalg.lstsq(X, itm_cashflows, rcond=None)

        continuation_values = (
            beta[0] + beta[1] * itm_swap_rates + beta[2] * itm_swap_rates**2
        )
        exercised_paths = (
            payoffs[col] > continuation_values.reindex(payoffs[col].index) * discount
        )
        cashflows.loc[exercised_paths] = payoffs[col].loc[exercised_paths]

    return sum(cashflows) / N
