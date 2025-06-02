import pandas as pd
import numpy as np
from scipy.optimize import minimize

from abc import ABC, abstractmethod
from dataclasses import dataclass

from _config import polynomial_classes


class algorithm(ABC):
    """Represente an LSM algorithm"""

    @abstractmethod
    def calibration(self): ...


@dataclass
class LSM_method_v1(algorithm):
    """
    A first implementation of the LSM algorithm.
    It allows for basis functions of different types, i.e. Power, Chebyshev, Hermite etc. Although no delta extensions.
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
        cashflows = payoffs[self.exercise_dates[-1]]

        for i in range(len(self.exercise_dates) - 2, -1, -1):
            t = self.exercise_dates[i]

            cashflows = cashflows * discount_factors[t]

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

        if self.exercise_dates[0] != 0:
            discount_times = [
                col for col in discount_factors.columns if col < self.exercise_dates[0]
            ]
            cumulative_discount = discount_factors[discount_times].prod(axis=1)
            cashflows = cashflows * cumulative_discount

        return sum(cashflows) / len(cashflows), fitted_basis_functions

    def estimation(
        self,
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        discount_factors: pd.DataFrame,
        fitted_basis_functions: dict,
    ):
        """
        Estimates option price given the underlying asset paths, discount factors and calibrated basis functions.

        underlying_asset_paths: Simulated paths of underlying asset.
        payoffs: Time t payoffs of the option given the underlying asset paths.
        discount_factors: Discount factors for each t to discount from t+1 to t.
        fitted_basis_functions: Fitted basis functions as returned by calibration above.
        """
        cashflows = pd.Series(0, index=payoffs.index, name="cashflows", dtype=float)
        for i in range(len(self.exercise_dates)):
            t = self.exercise_dates[i]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]

            if t == self.exercise_dates[-1]:
                cashflows.loc[itm_paths] = (
                    payoffs[t].loc[itm_paths] * discount.loc[itm_paths]
                )
                continue

            exercised_paths = self._exercise_evluation(
                t=t,
                fitted_basis_function=fitted_basis_functions[t],
                underlying_asset_values=itm_asset_paths,
                payoffs=payoffs,
            )

            if i > 0:
                cashflows.loc[exercised_paths] = (
                    payoffs[t].loc[exercised_paths] * discount.loc[exercised_paths]
                )
                discount = discount * discount_factors[t]
            else:
                cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]
                discount = discount_factors[self.exercise_dates[0]]

            payoffs.loc[exercised_paths, t:] = 0

        if self.exercise_dates[0] != 0:
            discount_times = [
                col for col in discount_factors.columns if col < self.exercise_dates[0]
            ]
            cumulative_discount = discount_factors[discount_times].prod(axis=1)
            cashflows = cashflows * cumulative_discount

        return sum(cashflows) / len(cashflows)


@dataclass
class LSM_method_v2(algorithm):
    """
    An implementation of the LSM algorithm using power polynomials and optionally delta regularization.
    """

    strike: float
    exercise_dates: list[float, ...]
    degree: int

    def _loss(
        self,
        beta: list[float],
        Y: pd.DataFrame,
        phi: pd.DataFrame,
        phi_prime,
        _lambda: float | None = None,
        Z: pd.DataFrame | None = None,
    ) -> float:

        pred = phi @ beta
        reg = phi_prime @ beta[1:]

        if _lambda == None and Z == None:
            return np.sum((Y - pred) ** 2)
        else:
            return np.sum((Y - pred) ** 2) + _lambda * np.sum((Z - reg) ** 2)

    def _regression(
        self,
        cashflows: pd.DataFrame,
        phi: np.array,
        phi_prime: np.array,
        _lambda: pd.DataFrame | None = None,
        Z: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Estimating beta coefficients for polynomial fit.
        """
        beta0 = np.zeros(phi.shape[1])

        if _lambda == None and Z == None:
            result = minimize(self._loss, beta0, args=(cashflows, phi, phi_prime))
        else:
            result = minimize(
                self._loss, beta0, args=(cashflows, phi, phi_prime, _lambda, Z)
            )

        beta = result.x
        return beta

    def _exercise_evluation(
        self,
        t: float,
        beta: np.array,
        payoffs: pd.DataFrame,
        phi: np.array,
        underlying_asset_paths: pd.DataFrame,
    ):
        """
        Calculates which paths to exercise.

        t: Time of decision.
        beta: Fitted beta coefficients to predict continuation values.
        payoffs: Time t payoffs used to compare with continuation values.
        phi: Phi function of the underlying
        underlying_asset_paths: Simulated paths of underlying asset.
        """
        continuation_values = phi @ beta

        return payoffs[t] > pd.Series(
            continuation_values, index=underlying_asset_paths.index
        ).reindex(payoffs[t].index)

    def calibration(
        self,
        method: str,
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        discount_factors: pd.DataFrame,
        accrual_factors: pd.DataFrame | None = None,
        a: float | None = None,
    ) -> pd.DataFrame:
        """
        Calibrates the regression beta coefficients by backwards recursion.
        Returns in-sample price estimate and coefficients.

        method: Specify which regression method to use.
        underlying_asset_paths: Simulated paths of underlying asset.
        payoffs: Time t payoffs of the option given the underlying asset paths.
        discount_factors: Discount factors for each t to discount from t+1 to t.
        accrual_factors: Simulated accrual factors for swap rate delta regularization. Optional argument.
        a: a parameter from the Vasicek model for short rate delta regularization. Optional argument.
        """
        if method not in ["classic", "stock_delta", "swap_delta", "short_rate_delta"]:
            raise ValueError(f"{method} is not a valid calibration method.")

        if method == "swap_delta" and accrual_factors is None:
            raise ValueError(
                f"No accrual_factors provided for swap rate delta regularization."
            )

        if method == "short_rate_delta" and a is None:
            raise ValueError(
                f"No 'a' parameter provided for short rate delta regularization."
            )

        degree = self.degree
        betas = {}

        cashflows = payoffs[self.exercise_dates[-1]]

        if method == "stock_delta":
            discounted_stock_paths = underlying_asset_paths[self.exercise_dates[-1]]

        elif method in "swap_delta":
            exercised_swap_rates = underlying_asset_paths[self.exercise_dates[-1]]
            discounted_accrual_factors = accrual_factors[self.exercise_dates[-1]]
        elif method == "short_rate_delta":
            tau = pd.Series(self.exercise_dates[-1], index=payoffs.index)

        for i in range(len(self.exercise_dates) - 2, -1, -1):
            t = self.exercise_dates[i]

            cashflows = cashflows * discount_factors[t]

            if method == "stock_delta":
                discounted_stock_paths = discounted_stock_paths * discount_factors[t]

            elif method == "swap_delta":
                discounted_accrual_factors = (
                    discounted_accrual_factors * discount_factors[t]
                )

                if i == len(self.exercise_dates) - 2:
                    accumulated_discount_factors = discount_factors[t]
                else:
                    accumulated_discount_factors = (
                        accumulated_discount_factors * discount_factors[t]
                    )

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            if itm_paths == []:
                t_plus_one = self.exercise_dates[i + 1]
                try:
                    betas[t] = betas[t_plus_one]
                except:  # WE STILL HAVE A PROBLEM HERE
                    raise Exception(
                        f"Unable to calculate regression coefficients for t = {t}"
                        f" due to to no ITM paths."
                    )

            itm_cashflows = cashflows[itm_paths].to_numpy()
            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]

            phi = np.vstack([itm_asset_paths**i for i in range(0, degree + 1)]).T
            phi_prime = np.vstack(
                [i * itm_asset_paths ** (i - 1) for i in range(1, degree + 1)]
            ).T  # maybe not do this if not using delta regularization

            if method == "stock_delta":
                Z = np.where(
                    itm_cashflows > 0,
                    -discounted_stock_paths[itm_paths] / itm_asset_paths,
                    0,
                )

            elif method == "swap_delta":
                Z = np.where(
                    itm_cashflows > 0,
                    discounted_accrual_factors[itm_paths]
                    - itm_cashflows / itm_asset_paths,
                    0,
                )

            elif method == "short_rate_delta":
                Z = np.where(
                    itm_cashflows > 0, np.exp(-a * (tau.loc[itm_paths] - t)), 0
                )
            else:
                Z = None

            if Z is not None:
                _lambda = (itm_cashflows @ itm_cashflows) / (Z @ Z)
            else:
                _lambda = None

            betas[t] = self._regression(
                cashflows=itm_cashflows,
                phi=phi,
                phi_prime=phi_prime,
                _lambda=_lambda,
                Z=Z,
            )

            exercised_paths = self._exercise_evluation(
                t=t,
                beta=betas[t],
                payoffs=payoffs,
                phi=phi,
                underlying_asset_paths=itm_asset_paths,
            )

            cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]

            if method == "stock_delta":
                discounted_stock_paths.loc[exercised_paths] = underlying_asset_paths[
                    t
                ].loc[exercised_paths]

            elif method == "swap_delta":
                accumulated_discount_factors.loc[exercised_paths] = 1
                exercised_swap_rates.loc[exercised_paths] = underlying_asset_paths[
                    t
                ].loc[exercised_paths]

            elif method == "short_rate_delta":
                tau.loc[exercised_paths] = t

        if self.exercise_dates[0] != 0:
            discount_times = [
                col for col in discount_factors.columns if col < self.exercise_dates[0]
            ]
            cumulative_discount = discount_factors[discount_times].prod(axis=1)
            cashflows = cashflows * cumulative_discount

        return sum(cashflows) / len(cashflows), betas

    def estimation(
        self,
        underlying_asset_paths: pd.DataFrame,
        payoffs: pd.DataFrame,
        discount_factors: pd.DataFrame,
        betas: dict,
    ):
        """
        Estimates option price given the underlying asset paths, discount factors and calibrated basis functions.

        underlying_asset_paths: Simulated paths of underlying asset.
        payoffs: Time t payoffs of the option given the underlying asset paths.
        discount_factors: Discount factors for each t to discount from t+1 to t.
        fitted_basis_functions: Fitted basis functions as returned by calibration above.
        """
        degree = self.degree
        cashflows = pd.Series(0, index=payoffs.index, name="cashflows", dtype=float)
        for i in range(len(self.exercise_dates)):
            t = self.exercise_dates[i]

            itm_paths = payoffs.index[payoffs[t] > 0].tolist()
            itm_asset_paths = underlying_asset_paths.loc[itm_paths, t]

            if t == self.exercise_dates[-1]:
                cashflows.loc[itm_paths] = payoffs[t].loc[itm_paths] * discount
                continue

            phi = np.vstack([itm_asset_paths**i for i in range(0, degree + 1)]).T

            exercised_paths = self._exercise_evluation(
                t=t,
                beta=betas[t],
                payoffs=payoffs,
                phi=phi,
                underlying_asset_paths=itm_asset_paths,
            )

            if i > 0:
                cashflows.loc[exercised_paths] = (
                    payoffs[t].loc[exercised_paths] * discount.loc[exercised_paths]
                )
                discount = discount * discount_factors[t]
            else:
                cashflows.loc[exercised_paths] = payoffs[t].loc[exercised_paths]
                discount = discount_factors[self.exercise_dates[0]]

            payoffs.loc[exercised_paths, t:] = 0

        if self.exercise_dates[0] != 0:
            discount_times = [
                col for col in discount_factors.columns if col < self.exercise_dates[0]
            ]
            cumulative_discount = discount_factors[discount_times].prod(axis=1)
            cashflows = cashflows * cumulative_discount

        return sum(cashflows) / len(cashflows)
