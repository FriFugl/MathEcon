import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from scipy.odr import polynomial

from _helpers import _calculate_swaption_payoffs
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
        self, swap_rates: pd.DataFrame, cashflows: pd.DataFrame
    ) -> np.ndarray:
        """
        Regression calculation performed when calibrating the LSM algorithm

        swap_rates: In-the-money swap rates used as regressor.
        cashflows: In-the-money cashflows used as response variable.
        """

        method = self.basis_function[0]
        degree = self.basis_function[1]

        if method == "polynomial":
            X = np.column_stack([swap_rates**i for i in range(degree + 1)])

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
        short_rates: pd.DataFrame,
        swap_rates: pd.DataFrame,
        accrual_factors: pd.DataFrame,
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

        payoffs = _calculate_swaption_payoffs(
            swap_rates=swap_rates, accrual_factors=accrual_factors, strike=self.strike
        )
        discount_factors = _short_rate_to_discount_factors(short_rates=short_rates)

        coefficients = {}
        cashflows = (
            payoffs[self.exercise_dates[-1]] * discount_factors[self.exercise_dates[-2]]
        )
        for i in range(len(self.exercise_dates) - 2, 0, -1):
            t = self.exercise_dates[i]
            delta = self.exercise_dates[i] - self.exercise_dates[i - 1]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            if itm_paths == []:
                raise Exception(
                    f"Unable to calculate regression coefficients for t = {t}"
                    f" due to to no ITM paths."
                )

            itm_swap_rates = swap_rates.loc[itm_paths, t]
            itm_cashflows = cashflows[itm_paths].to_numpy()

            coefficients[t] = self._regression(
                swap_rates=itm_swap_rates, cashflows=itm_cashflows
            )

            exercised_paths = self._exercise_evluation(
                t=t,
                coefficients=coefficients,
                swap_rates=itm_swap_rates,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]
            cashflows = cashflows * discount_factors[t - delta]

        return sum(cashflows) / len(cashflows), coefficients

    def estimation(
        self,
        short_rates: pd.DataFrame,
        swap_rates: pd.DataFrame,
        accrual_factors: pd.DataFrame,
        coefficients: dict,
    ):

        payoffs = _calculate_swaption_payoffs(
            swap_rates=swap_rates, accrual_factors=accrual_factors, strike=self.strike
        )

        discount_factors = _short_rate_to_discount_factors(short_rates=short_rates)

        discount = discount_factors[0]
        cashflows = pd.Series(0, index=payoffs.index, name="cashflows", dtype=float)
        for i in range(1, len(self.exercise_dates[1 : len(self.exercise_dates)])):
            t = self.exercise_dates[i]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            itm_swap_rates = swap_rates.loc[itm_paths, t]

            exercised_paths = self._exercise_evluation(
                t=t,
                coefficients=coefficients,
                swap_rates=itm_swap_rates,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths] * discount
            payoffs.loc[exercised_paths, t:] = 0

            discount = discount * discount_factors[t]

        return sum(cashflows) / len(cashflows)
