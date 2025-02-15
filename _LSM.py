import pandas as pd
import numpy as np

from _calculate_swap_and_swaption_details import _calculate_swaption_payoffs


def _LSM_method(
    short_rates: pd.DataFrame,
    swap_rates: pd.DataFrame,
    accrual_factors: pd.DataFrame,
    strike: float,
    exercise_dates: list[float],
) -> pd.DataFrame:
    """
    Perform classic Longstaff-Schwartz algorithm for a swaption.

    short_rates: Simulated short rates.
    swap_rates: Simulated swap rates corresponding to the short_rates.
    accrual_factors: Simulated swap rates corresponding to the short_rates.
    strike: Strike of Swaption.
    exercise_dates: Swaption exercise dates.
    N: Number of simulated tracjetories.
    """

    payoffs = _calculate_swaption_payoffs(
        swap_rates=swap_rates, accrual_factors=accrual_factors, strike=strike
    )
    discounts = np.exp(
        -short_rates.loc[:, exercise_dates[:-1]].iloc[0].values
        * np.diff(exercise_dates)
    )

    cashflows = payoffs[exercise_dates[-1]] * discounts[-1]
    for i in range(len(exercise_dates) - 2, -1, -1):
        t = exercise_dates[i]

        itm_paths = payoffs.index[payoffs[t] > 0].tolist()
        if itm_paths == []:
            cashflows = cashflows * discounts[i - 1]
            continue

        itm_swap_rates = swap_rates.loc[itm_paths, t]
        itm_cashflows = cashflows[itm_paths].to_numpy()

        X = np.column_stack(
            (np.ones(len(itm_swap_rates)), itm_swap_rates, itm_swap_rates**2)
        )
        beta, residuals, rank, s = np.linalg.lstsq(X, itm_cashflows, rcond=None)

        continuation_values = (
            beta[0] + beta[1] * itm_swap_rates + beta[2] * itm_swap_rates**2
        )
        exercised_paths = payoffs[t] > continuation_values.reindex(payoffs[t].index)

        cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]

        if t != 0:
            cashflows = cashflows * discounts[i - 1]

    return sum(cashflows) / len(cashflows)
