import pandas as pd
import numpy as np

from _calculate_swap_and_swaption_details import _compute_swap_rate_and_accrual_factor

from abc import ABC, abstractmethod
from dataclasses import dataclass


class StochasticProcess(ABC):
    """Represente a Stochastic process"""

    @abstractmethod
    def simulate(self): ...


@dataclass
class VasicekModel(StochasticProcess):
    """
    The Vasicek model which can be used to simulate the short rate.
    """

    a: float
    b: float
    sigma: float

    def simulate(
        self, r_0: float, T: int, M: int, N: int, method: str, seed: int = None
    ) -> pd.DataFrame:
        """
        r_0: Initial short rate.
        T: Length of time grid.
        M: Time discretization.
        method: Method of simulation.
        N: Number of simulated paths.
        """
        if seed is not None:
            np.random.seed(seed)

        delta = T / M

        r = np.zeros([N, M + 1])
        r[:, 0] = r_0

        z = np.random.standard_normal((N, M))

        if method == "exact":
            for m in range(1, M + 1):
                r[:, m] = (
                    r[:, m - 1] * np.exp(-self.a * delta)
                    + (self.b / self.a) * (1 - np.exp(-self.a * delta))
                    + self.sigma
                    * np.sqrt((1 - np.exp(-2 * self.a * delta)) / (2 * self.a))
                    * z[:, m - 1]
                )
        if method == "euler":
            for m in range(1, M + 1):
                r[:, m] = (
                    (r[:, m - 1])
                    + (self.b - self.a * r[:, m - 1])
                    + self.sigma * np.sqrt(delta) * z[:, m - 1]
                )

        short_rates = pd.DataFrame(
            r, index=[i for i in range(1, N + 1)], columns=np.linspace(0, T, M + 1)
        )

        return short_rates

    def price_zcb(
        self,
        short_rates: pd.DataFrame,
        t: float,
        maturities: list[float],
    ) -> pd.DataFrame:
        """
        Convert simulated short rates in Vasicek model to ZCB prices.

        short_rates: Simulated short rates in Vasicek model.
        t: Present time.
        maturities: List of maturity dates.
        """

        B = []
        A = []
        for T in maturities:
            b_t_T = (1 / self.a) * (1 - np.exp(-self.a * (T - t)))
            a_t_T = ((b_t_T - T + t) * (self.a * self.b - 0.5 * (self.sigma**2))) / (
                self.a**2
            ) - ((self.sigma**2) * (b_t_T**2)) / (4 * self.a)

            B.append(b_t_T)
            A.append(a_t_T)

        r = short_rates.loc[:, t]

        num_rows = short_rates.shape[0]
        zcb_prices = pd.DataFrame({t: np.ones(num_rows)}, index=range(1, num_rows + 1))

        for i in range(len(maturities)):
            zcb_prices[maturities[i]] = np.exp(A[i] - B[i] * r)

        return zcb_prices

    def swap_rate(
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
