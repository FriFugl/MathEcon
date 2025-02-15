import pandas as pd
import numpy as np


def _compute_swap_rate_and_accrual_factor(
    zcb_prices: pd.DataFrame,
    start: float,
    maturity: float,
    alpha: float,
) -> pd.DataFrame:
    """
    Computes the par swap rate and accrual factor.

    zcb_prices: ZCB prices in a pandas dataframe starting from p(0,0) to some maturity.
    start: First resettlement.
    maturity: Maturity of swap.
    alpha: Time interval between swap fixings in years.
    """

    accrual_factors = alpha * zcb_prices.loc[:, start:maturity].sum(axis=1)
    swap_rates = (1 - zcb_prices[maturity]) / accrual_factors

    return swap_rates, accrual_factors


def _calculate_swaption_payoffs(
    swap_rates: pd.DataFrame, accrual_factors: pd.DataFrame, strike, payer: bool = True
) -> pd.DataFrame:
    """
    Function to calculate swaption payoff.


    swap_rates: Dataframe with swap rates at time points.
    accrual_factors: Dataframe with accrual factors at time points.
    strike: Strike rate of swaption.
    payer: True for payer swaption, False for receiver swaption. Default is True.
    """

    if payer:
        return accrual_factors * np.maximum(swap_rates - strike, 0)
    else:
        return accrual_factors * np.minimum(strike - swap_rates, 0)
