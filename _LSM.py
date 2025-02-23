import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from _config import polynomial_classes


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
    exercise_dates: list[float, ...]
    basis_function: tuple[str, int]

    def _regression(
        self, underlying_asset_values: pd.DataFrame, cashflows: pd.DataFrame
    ) -> np.ndarray:
        """
        Regression calculation performed when calibrating the LSM algorithm

        underlying_asset_values: In-the-money assets.
        cashflows: In-the-money cashflows used as response variable.
        """

        polynomial_type = self.basis_function[0]
        degree = self.basis_function[1]

        if polynomial_type not in polynomial_classes:
            raise ValueError(
                f"{polynomial_type} is an invalid polynomial type. "
                "Must be one of: " + ", ".join(polynomial_classes.keys())
            )

        polynomial = polynomial_classes[polynomial_type](
            (i for i in range(1, degree + 1))
        )

        return polynomial.fit(x=underlying_asset_values, y=cashflows, deg=degree)

    def _exercise_evluation(
        self,
        t: float,
        fitted_basis_function: np.polynomial.polynomial,
        underlying_asset_values: pd.DataFrame,
        payoffs: pd.DataFrame,
    ):
        """
        Calculates which paths to exercise.

        t: Time of decision.
        fitted_basis_function: Fitted np.polynomial.polynomial to predict continuation values.
        underlying_asset_values: In-the-money assets used to estimate continuation value.
        payoffs: Time t payoffs used to compare with continuation values.
        """
        continuation_values = fitted_basis_function(underlying_asset_values)

        return payoffs[t] > pd.Series(
            continuation_values, index=underlying_asset_values.index
        ).reindex(payoffs[t].index)

    def calibration(
        self,
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        discount_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calibrates the regression coefficients by backwards recursion.
        Returns in-sample price estimate and coefficients.

        underlying_asset_paths: Simulated paths of underlying asset.
        payoffs: Time t payoffs of the option given the underlying asset paths.
        discount_factors: Discount factors for each t to discount from t+1 to t.
        """

        fitted_basis_functions = {}
        cashflows = (
            payoffs[self.exercise_dates[-1]] * discount_factors[self.exercise_dates[-2]]
        )

        for i in range(len(self.exercise_dates) - 2, 0, -1):
            t = self.exercise_dates[i]
            t_minus_one = self.exercise_dates[i - 1]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            if itm_paths == []:
                t_plus_one = self.exercise_dates[i + 1]
                try:
                    fitted_basis_functions[t] = fitted_basis_functions[t_plus_one]
                except:  # WE STILL HAVE A PROBLEM HERE
                    raise Exception(
                        f"Unable to calculate regression coefficients for t = {t}"
                        f" due to to no ITM paths."
                    )
            itm_cashflows = cashflows[itm_paths].to_numpy()
            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]

            fitted_basis_functions[t] = self._regression(
                underlying_asset_values=itm_asset_paths, cashflows=itm_cashflows
            )

            exercised_paths = self._exercise_evluation(
                t=t,
                fitted_basis_function=fitted_basis_functions[t],
                underlying_asset_values=itm_asset_paths,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]
            cashflows = cashflows * discount_factors[t_minus_one]

        return sum(cashflows) / len(cashflows), fitted_basis_functions

    def estimation(
        self,
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        discount_factors: pd.DataFrame,
        fitted_basis_functions: dict,
    ):

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
                fitted_basis_function=fitted_basis_functions[t],
                underlying_asset_values=itm_asset_paths,
                payoffs=payoffs,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths] * discount
            payoffs.loc[exercised_paths, t:] = 0

            discount = discount * discount_factors[t]

        return sum(cashflows) / len(cashflows)
