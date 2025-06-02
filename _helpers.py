import pandas as pd
import numpy as np
from scipy.special import ndtr


def _short_rate_to_discount_factors(short_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Computes discount factors from dataframe of short_rates.

    short_rates: pandas dataframe of short_rates in which the column names are time points.
    """

    return pd.DataFrame(
        np.exp(-short_rates.iloc[:, :-1] * np.diff(short_rates.columns.to_numpy())),
        index=short_rates.index,
        columns=short_rates.columns[:-1],
    )


def _calculate_swap_rate_and_accrual_factor(
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
    swap_rates: pd.DataFrame,
    accrual_factors: pd.DataFrame,
    strike: float,
    payer: bool = True,
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


def _calculate_option_payoffs(
    stock_paths: pd.DataFrame, strike: float, call: bool = False
) -> pd.DataFrame:
    """
    Function to calculate put option payoffs.

    stock_paths: Dataframe with stock paths at time points.
    strike: Strike rate of swaption.
    call: True for call option. False for put option. Default is False.
    """
    if call:
        return np.maximum(stock_paths - strike, 0)
    else:
        return np.maximum(strike - stock_paths, 0)

def _black_swaption_price(swap_rate: float,
                          accrual_factor: float,
                          strike: float,
                          sigma: float,
                          T_n: float,
                          type: str = 'payer') -> float:
    """
    Function to calculate European swaption prices using the Black-76 formula.

    swap_rate: Swap rate.
    accrual_factor: Acrual factor.
    strike: Strike of the swaption contract.
    sigma: Black Implied volatility.
    T_n: Start of payment of Swap.
    type: Payer/Receiver swap type.
    """

    d1 = (np.log(swap_rate / strike) + 0.5 * (sigma**2) * T_n) / (sigma*np.sqrt(T_n))
    d2 = d1 - sigma * np.sqrt(T_n)

    if type == 'payer':
        return accrual_factor * (swap_rate*ndtr(d1) - strike * ndtr(d2))
    elif type == 'receiver':
        return  accrual_factor * (strike * ndtr(-d2) - swap_rate*ndtr(-d1))
