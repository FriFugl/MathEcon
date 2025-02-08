import pandas as pd
import numpy as np

def _simulate_vasicek_short_rate(r_0: float,
                                 a: float,
                                 b: float,
                                 sigma: float,
                                 T: int,
                                 M: int,
                                 method: str,
                                 N: int
                                 ) -> pd.DataFrame:

    """
    Simulates the Vasicek short rate on a time grid:

    r_0: Starting short rate
    a: Speed of reversion.
    b: Long term mean.
    sigma: Volatility.
    T: Length of time grid.
    M: Time discretization.
    method: Method of simulation.
    N: Number of simulated paths.
    """

    delta = T/M

    r = np.zeros((N, M+1))
    r[:,0] = r_0

    z = np.random.randn(N, M+1)

    if method == 'exact':
        for m in range(M):
            r[:,m+1] = (r[:,m] * np.exp(-a*delta) + (b/a) * (1-np.exp(-a*delta))
                    + ((sigma**2)/a) * (1-np.exp(-2*a*delta))) * z[:,m]

    short_rates = pd.DataFrame(r, index=[i for i in range(1,N+1)], columns=np.linspace(0, T, M+1))

    return short_rates

def _short_rate_to_zcb_prices_vasicek(short_rates: pd.DataFrame,
                             t: float,
                             maturities: list[float],
                             a: float,
                             b: float,
                             sigma: float
                             ) -> pd.DataFrame:

    """
    Convert simulated short rates in Vasicek model to ZCB prices.

    short_rates: Simulated short rates in Vasicek model.
    t: Present time.
    maturities: List of maturity dates.
    a: Speed of reversion.
    b: Long term mean.
    sigma: Volatility.
    """

    B = []
    A = []
    for T in maturities:
        b_t_T = (1/a) * (1 - np.exp(-a*(T-t)))
        a_t_T = ((b_t_T - T + t) * (a*b - 0.5*sigma**2))/(a**2) - ((sigma**2)*(b_t_T**2))/(4*a)

        B.append(b_t_T)
        A.append(a_t_T)

    r = short_rates.loc[:, t]

    num_rows = short_rates.shape[0]
    zcb_prices = pd.DataFrame({maturities[0]: np.ones(num_rows)},index=range(1, num_rows + 1))

    for i in maturities[1:]:
        zcb_prices[maturities[i]] = np.exp(A[i]-B[i]*r)

    return zcb_prices
