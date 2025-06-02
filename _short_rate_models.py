import pandas as pd
import numpy as np

from _helpers import _calculate_swap_rate_and_accrual_factor
from _processes import StochasticProcess

from dataclasses import dataclass


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
        N: Number of simulated paths.
        method: Method of simulation.
        seed: Keyword argument to set seed.
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
            delta_sqrt = np.sqrt(delta)
            for m in range(1, M + 1):
                r[:, m] = (
                    (r[:, m - 1])
                    + (self.b - self.a * r[:, m - 1]) * delta
                    + self.sigma * delta_sqrt * z[:, m - 1]
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
        zcb_prices = pd.DataFrame(index=range(1, num_rows + 1))

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
        alpha: Time between fixed payments.
        """
        swap_rates = {}
        accrual_factors = {}

        for date in entry_dates:
            start_date = min(
                i for i in np.arange(0, expiry + alpha, alpha) if i > date + 1e-5
            )
            swap_annuities = np.arange(start_date, expiry + alpha, alpha)

            zcb_prices = self.price_zcb(
                short_rates=short_rate, t=date, maturities=swap_annuities
            )

            R, S = _calculate_swap_rate_and_accrual_factor(
                zcb_prices=zcb_prices, start=start_date, maturity=expiry, alpha=alpha
            )

            swap_rates[date] = R
            accrual_factors[date] = S

        swap_rates = pd.DataFrame(swap_rates)
        accrual_factors = pd.DataFrame(accrual_factors)

        return swap_rates, accrual_factors


@dataclass
class GaussianModel(StochasticProcess):
    """
    The 2-Dimensional Guassian model which can be used to simulate the short rate.
    """

    a: float
    b: float
    sigma: float
    eta: float
    rho: float
    instant_forward_rates: dict[float:float]

    def simulate(
        self, T: int, M: int, N: int, method: str, seed: int = None
    ) -> pd.DataFrame:
        """
        r_0: Initial short rate.
        T: Length of time grid.
        M: Time discretization.
        N: Number of simulated paths.
        method: Method of simulation.
        seed: Keyword argument to set seed.
        """
        if seed is not None:
            np.random.seed(seed)

        delta = T / M

        forward_rate_times = list(self.instant_forward_rates.keys())
        forward_rates = pd.Series(
            [
                (
                    self.instant_forward_rates[i * delta]
                    if i * delta in forward_rate_times
                    else np.nan
                )
                for i in range(M + 1)
            ],
            index=[i * (T / M) for i in range(M + 1)],
        )
        interpolated_forward_rates = forward_rates.interpolate().ffill().bfill()

        varphi = [
            interpolated_forward_rates[i * delta]
            + ((self.sigma**2) / (2 * self.a**2))
            * (1 - np.exp(-self.a * i * delta)) ** 2
            + ((self.eta**2) / (2 * self.b**2)) * (1 - np.exp(-self.b * i * delta)) ** 2
            + ((self.rho * self.sigma * self.eta) / (self.a * self.b))
            * (1 - np.exp(-self.a * i * delta))
            * (1 - np.exp(-self.b * i * delta))
            for i in range(M + 1)
        ]

        r = np.zeros([N, M + 1])
        r[:, 0] = interpolated_forward_rates[0]

        x = np.zeros([N, M + 1])
        x[:, 0] = 0

        y = np.zeros([N, M + 1])
        y[:, 0] = 0

        w_1 = np.random.standard_normal((N, M))
        w_2 = np.random.standard_normal((N, M))

        if method == "euler":
            delta_sqrt = np.sqrt(delta)
            rho_2 = np.sqrt(1 - self.rho**2)
            for m in range(1, M + 1):
                x[:, m] = (
                    x[:, m - 1]
                    - self.a * x[:, m - 1] * delta
                    + self.sigma * delta_sqrt * w_1[:, m - 1]
                )

                y[:, m] = (
                    y[:, m - 1]
                    - self.b * y[:, m - 1] * delta
                    + delta_sqrt
                    * (
                        self.eta * self.rho * w_1[:, m - 1]
                        + self.eta * rho_2 * w_2[:, m - 1]
                    )
                )

                r[:, m] = x[:, m] + y[:, m] + varphi[m]

        short_rates = pd.DataFrame(
            r, index=[i for i in range(1, N + 1)], columns=np.linspace(0, T, M + 1)
        )
        x_paths = pd.DataFrame(
            x, index=[i for i in range(1, N + 1)], columns=np.linspace(0, T, M + 1)
        )
        y_paths = pd.DataFrame(
            y, index=[i for i in range(1, N + 1)], columns=np.linspace(0, T, M + 1)
        )

        varphi = dict(zip([i * (T / M) for i in range(M + 1)], varphi))

        return short_rates, x_paths, y_paths, varphi

    def price_zcb(
        self,
        x_paths: pd.DataFrame,
        y_paths: pd.DataFrame,
        varphi: list[float],
        t: float,
        maturities: list[float],
    ) -> pd.DataFrame:
        """
        short_rates: Simulated short rates in Vasicek model.
        t: Present time.
        maturities: List of maturity dates.
        """

        V = []
        for T in maturities:
            V_t_T = (
                ((self.sigma**2) / (self.a**2))
                * (
                    T
                    - t
                    + (2 / self.a) * np.exp(-self.a * (T - t))
                    - (1 / (2 * self.a)) * np.exp(-2 * self.a * (T - t))
                    - (3 / (2 * self.a))
                )
                + ((self.eta**2) / (self.b**2))
                * (
                    T
                    - t
                    + (2 / self.b) * np.exp(-self.b * (T - t))
                    - (1 / (2 * self.b)) * np.exp(-2 * self.b * (T - t))
                    - (3 / (2 * self.b))
                )
                + ((2 * self.rho * self.sigma * self.eta) / (self.a * self.b))
                * (
                    T
                    - t
                    + (np.exp(-self.a * (T - t)) - 1) / self.a
                    + (np.exp(-self.b * (T - t)) - 1) / self.b
                    - (np.exp(-(self.a + self.b) * (T - t)) - 1) / (self.a + self.b)
                )
            )

            V.append(V_t_T)

        x = x_paths.loc[:, t]
        y = y_paths.loc[:, t]

        num_rows = x_paths.shape[0]
        zcb_prices = pd.DataFrame(index=range(1, num_rows + 1))

        for i in range(len(maturities)):
            T = maturities[i]
            if i == 0:
                varphi_integral = varphi[T] * (T - t)
            else:
                varphi_integral += varphi[T] * (T - maturities[i - 1])

            zcb_prices[maturities[i]] = np.exp(
                -varphi_integral
                - ((1 - np.exp(-self.a * (T - t))) / self.a) * x
                - ((1 - np.exp(-self.b * (T - t))) / self.b) * y
                + 0.5 * V[i]
            )

        return zcb_prices

    def swap_rate(
        self,
        x_paths: pd.DataFrame,
        y_paths: pd.DataFrame,
        varphi: list[float],
        entry_dates: float,
        expiry: float,
        alpha: float,
    ) -> pd.DataFrame:
        """
        Convert short rates in a Vasicek model to swap rates.

        short_rates: Simulated short rates.
        entry_dates: List of dates for which swap rates is entered. Each date is equivalent to T_{n-1}.
        expiry: Years until swap expiry. Equivalent to T_N.
        alpha: Time between fixed payments.
        """
        swap_rates = {}
        accrual_factors = {}

        for date in entry_dates:
            start_date = min(i for i in np.arange(0, expiry, alpha) if i > date + 1e-5)
            swap_annuities = np.arange(start_date, expiry + alpha, alpha)

            zcb_prices = self.price_zcb(
                x_paths=x_paths,
                y_paths=y_paths,
                varphi=varphi,
                t=date,
                maturities=swap_annuities,
            )

            R, S = _calculate_swap_rate_and_accrual_factor(
                zcb_prices=zcb_prices, start=start_date, maturity=expiry, alpha=alpha
            )

            swap_rates[date] = R
            accrual_factors[date] = S

        swap_rates = pd.DataFrame(swap_rates)
        accrual_factors = pd.DataFrame(accrual_factors)

        return swap_rates, accrual_factors
