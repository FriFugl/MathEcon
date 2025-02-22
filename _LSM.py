import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from _helpers import _short_rate_to_discount_factors



class algorithm(ABC):
    """Represente a Monte Carlo algorithm"""

    @abstractmethod
    def calibration(self): ...


@dataclass
class LSM_method(algorithm):
    """
    The classic Longstaff Schwarz Monte-Carlo method.
    """

    strike: float
    exercise_dates: float
    basis_function: tuple

    def _regression(
        self, underlying_asset: pd.DataFrame, cashflows: pd.DataFrame
    ) -> np.ndarray:
        """
        Regression calculation performed when calibrating the LSM algorithm

        swap_rates: In-the-money swap rates used as regressor.
        cashflows: In-the-money cashflows used as response variable.
        """

        method = self.basis_function[0]
        degree = self.basis_function[1]

        if method == "polynomial":
            X = np.column_stack([underlying_asset**i for i in range(degree + 1)])

        beta, residuals, rank, s = np.linalg.lstsq(X, cashflows, rcond=None)

        return beta

    def _exercise_evluation(
        self,
        t: float,
        coefficients: dict,
        swap_rates: pd.DataFrame,
        payoffs: pd.DataFrame,
    ):
        """
        Calculates which paths to exercise.

        t: Time of decision.
        coefficients: Dictionary of prediction coefficients.
        swap_rates: In-the-money swap rates used to estimate continuation value.
        payoffs: Time t payoffs used to compare with continuation values.
        """

        method = self.basis_function[0]
        degree = self.basis_function[1]

        if method == "polynomial":
            continuation_values = sum(
                [coefficients[t][i] * swap_rates**i for i in range(degree + 1)]
            )

        return payoffs[t] > continuation_values.reindex(payoffs[t].index)

    def calibration(
        self,
        short_rate: Union[pd.DataFrame, float, int],
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calibrates the regression coefficients by backwards recursion.
        Returns in-sample price estimate and coefficients.

        short_rates: Simulated short rates.
        swap_rates: Simulated swap rates corresponding to the short_rates.
        accrual_factors: Simulated swap rates corresponding to the short_rates.
        strike: Strike of Swaption.
        exercise_dates: Swaption exercise dates.
        N: Number of simulated tracjetories.
        """

        if isinstance(short_rate, (float, int)):
            short_rate = pd.DataFrame(short_rate,
                                      index=payoffs.index,
                                      columns=payoffs.columns
                                      )

        discount_factors = _short_rate_to_discount_factors(short_rates=short_rate)

        coefficients = {}
        cashflows = (
            payoffs[self.exercise_dates[-1]] * discount_factors[self.exercise_dates[-2]]
        )

        for i in range(len(self.exercise_dates) - 2, 0, -1):
            t = self.exercise_dates[i]
            t_minus_one =  self.exercise_dates[i-1]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            if itm_paths == []:
                t_plus_one = self.exercise_dates[i+1]
                try:
                    coefficients[t] = coefficients[t_plus_one]
                except: #WE STILL HAVE A PROBLEM HERE
                    raise Exception(
                        f"Unable to calculate regression coefficients for t = {t}"
                        f" due to to no ITM paths."
                    )

            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]
            itm_cashflows = cashflows[itm_paths].to_numpy()

            coefficients[t] = self._regression(
                underlying_asset=itm_asset_paths, cashflows=itm_cashflows
            )

            exercised_paths = self._exercise_evluation(
                t=t,
                coefficients=coefficients,
                swap_rates=itm_asset_paths,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]
            cashflows = cashflows * discount_factors[t_minus_one]

        return sum(cashflows) / len(cashflows), coefficients

    def estimation(
        self,
        short_rate: Union[pd.DataFrame, float, int],
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        coefficients: dict,
    ):
        if isinstance(short_rate, (float, int)):
            short_rate = pd.DataFrame(short_rate,
                                      index=payoffs.index,
                                      columns=payoffs.columns
                                      )

        discount_factors = _short_rate_to_discount_factors(short_rates=short_rate)

        discount = discount_factors[0]
        cashflows = pd.Series(0, index=payoffs.index, name="cashflows", dtype=float)
        for i in range(1, len(self.exercise_dates)):
            t = self.exercise_dates[i]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]

            if t == self.exercise_dates[-1]:
                cashflows.loc[itm_paths] = payoffs[t].loc[itm_paths] * discount
                continue

            exercised_paths = self._exercise_evluation(
                t=t,
                coefficients=coefficients,
                swap_rates=itm_asset_paths,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths] * discount
            payoffs.loc[exercised_paths, t:] = 0

            discount = discount * discount_factors[t]

        return sum(cashflows) / len(cashflows)
