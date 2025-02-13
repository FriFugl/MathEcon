import pandas as pd
import numpy as np

from _calculate_swap_and_swaption_details import _compute_swap_rate_and_accrual_factor

from abc import ABC, abstractmethod
from dataclasses import dataclass

class StochasticProcess(ABC):
    """Represente a Stochastic process"""

    @abstractmethod
    def simulate(self):
        ...

@dataclass
class VasicekModel(StochasticProcess):
    """
    The Vasicek model which can be used to simulate the short rate.
    """

    a: float
    b: float
    sigma: float

    def simulate(
            self, r_0: float, T: int, M: int, method: str, N: int
    ) -> pd.DataFrame:
        """
        r_0: Initial short rate.
        T: Length of time grid.
        M: Time discretization.
        method: Method of simulation.
        N: Number of simulated paths.
        """

        delta = T / M

        r = np.zeros((N, M + 1))
        r[:, 0] = r_0

        z = np.random.randn(N, M)

        if method == "exact":
            for m in range(M):
                r[:, m + 1] = (
                                      r[:, m] * np.exp(-self.a * delta)
                                      + (self.b / self.a) * (1 - np.exp(-self.a * delta))
                                      + self.sigma * np.sqrt((1 - np.exp(-2 * self.a * delta)) / (2*self.a))
                              ) * z[:, m]

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
            a_t_T = ((b_t_T - T + t) * (self.a * self.b - 0.5 * (self.sigma**2))) / (self.a**2) - (
                (self.sigma**2) * (b_t_T**2)
            ) / (4 * self.a)

            B.append(b_t_T)
            A.append(a_t_T)

        r = short_rates.loc[:, t]

        num_rows = short_rates.shape[0]
        zcb_prices = pd.DataFrame({t: np.ones(num_rows)}, index=range(1, num_rows + 1))

        for i in range(len(maturities)):
            zcb_prices[maturities[i]] = np.exp(A[i] - B[i] * r)

        return zcb_prices


    def swap_rate(
        self,
        short_rate: pd.DataFrame,
        expiry: float,
        exercise_dates: list,
    ) -> pd.DataFrame:
        """
        Convert short rates in a Vasicek model to swap rates.

        short_rates: Simulated short rates.
        expiry: Years until expiry. Equivalent to T_N.
        exercise_dates: List of dates for which swap rates should be calculated. Each date is Equivalent to T_n.
        """
        swap_rates = {}
        accrual_factors = {}

        for exercise_date in exercise_dates[:-1]:
            start_date = exercise_date + 1

            swap_maturities = [i for i in range(start_date, expiry + 1)]

            zcb_prices = self.price_zcb(
                short_rates=short_rate,
                t=exercise_date,
                maturities=swap_maturities
            )

            R, S = _compute_swap_rate_and_accrual_factor(
                zcb_prices=zcb_prices, start=start_date, maturity=expiry
            )

            swap_rates[exercise_date] = R
            accrual_factors[exercise_date] = S

        swap_rates = pd.DataFrame(swap_rates)
        accrual_factors = pd.DataFrame(accrual_factors)

        return swap_rates, accrual_factors