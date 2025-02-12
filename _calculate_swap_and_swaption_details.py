import pandas as pd
import numpy as np


def _compute_swap_rate_and_accrual_factor(
    zcb_prices: pd.DataFrame, start: float, maturity: float
) -> pd.DataFrame:
    """
    Computes the par swap rate and accrual factor.

    zcb_prices: ZCB prices in a pandas dataframe starting from p(0,0) to some maturity.
    start: First resettlement.
    maturity: Maturity of swap.
    """

    start_index = zcb_prices.columns.get_loc(start)
    maturity_index = zcb_prices.columns.get_loc(maturity)

    accrual_factors = zcb_prices.iloc[:, 1 : maturity_index + 1].sum(axis=1)
    swap_rate = (zcb_prices[start] - zcb_prices[maturity]) / accrual_factors

    return swap_rate, accrual_factors


def _calculate_swaption_payoffs(
    swap_rates: pd.DataFrame, accrual_factors: pd.DataFrame, strike
) -> pd.DataFrame:
    return accrual_factors * np.maximum(swap_rates - strike, 0)
