import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from _calculate_swap_and_swaption_details import _calculate_swaption_payoffs

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

    def calibration(self,
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
        discounts = pd.DataFrame(
            np.exp(-short_rates.iloc[:, :-1] * np.diff(short_rates.columns.to_numpy())),
            index=short_rates.index,
            columns=short_rates.columns[:-1]
        )

        coefficients = {}
        cashflows = payoffs[self.exercise_dates[-1]] * discounts[self.exercise_dates[-1]]
        for i in range(len(self.exercise_dates) - 2, -1, -1):
            t = self.exercise_dates[i]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            if itm_paths == []:
                cashflows = cashflows * discounts[t]
                """MAYBE CREATING BETA COEFFICIENTS IS NECESSARY HERE"""
                continue

            itm_swap_rates = swap_rates.loc[itm_paths, t]
            itm_cashflows = cashflows[itm_paths].to_numpy()

            X = np.column_stack(
                (np.ones(len(itm_swap_rates)), itm_swap_rates, itm_swap_rates**2)
            )
            beta, residuals, rank, s = np.linalg.lstsq(X, itm_cashflows, rcond=None)

            coefficients[t] = beta

            continuation_values = (
                beta[0] + beta[1] * itm_swap_rates + beta[2] * itm_swap_rates**2
            )
            exercised_paths = payoffs[t] > continuation_values.reindex(payoffs[t].index)

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]

            if t != 0:
                cashflows = cashflows * discounts[t]

        return sum(cashflows) / len(cashflows), pd.DataFrame(coefficients)
