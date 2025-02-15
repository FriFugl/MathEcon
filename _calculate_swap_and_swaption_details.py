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

    def _swap_rates_from_zcb_pr(
        self, short_rate: pd.DataFrame, entry_dates: float, expiry: float, alpha: float
    ) -> pd.DataFrame:
        """
        Convert short rates in a Vasicek model to swap rates.

        short_rates: Simulated short rates.
        entry_dates: List of dates for which swap rates is entered. Each date is equivalent to T_{n-1}.
        expiry: Years until swap expiry. Equivalent to T_N.
        """
        swap_rates = {}
        accrual_factors = {}

        for date in entry_dates:
            start_date = min(i for i in np.arange(0, expiry, alpha) if i > date + 1e-5)
            swap_annuities = np.arange(start_date, expiry + alpha, alpha)

            zcb_prices = self.price_zcb(
                short_rates=short_rate, t=date, maturities=swap_annuities
            )

            R, S = _compute_swap_rate_and_accrual_factor(
                zcb_prices=zcb_prices, start=start_date, maturity=expiry, alpha=alpha
            )

            swap_rates[date] = R
            accrual_factors[date] = S

        swap_rates = pd.DataFrame(swap_rates)
        accrual_factors = pd.DataFrame(accrual_factors)

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
