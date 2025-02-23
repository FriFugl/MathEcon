import pandas as pd
import numpy as np

from _processes import StochasticProcess

from dataclasses import dataclass


@dataclass
class GeometricBrownianMotion(StochasticProcess):
    """
    Standard Geometric Brownian Motion
    """

    r: float
    sigma: float

    def simulate(
        self, s_0: float, T: int, M: int, N: int, seed: int = None
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

        s = np.zeros([N, M + 1])
        s[:, 0] = s_0

        z = np.random.standard_normal((N, M))

        for m in range(1, M + 1):
            s[:, m] = s[:, m - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * delta
                + self.sigma * np.sqrt(delta) * z[:, m - 1]
            )

        stock_paths = pd.DataFrame(
            s, index=[i for i in range(1, N + 1)], columns=np.linspace(0, T, M + 1)
        )

        return stock_paths
