import pandas as pd

def _compute_swap_rate_and_accrual_factor(zcb_prices: pd.DataFrame, start: float, maturity: float) -> pd.DataFrame:

    """
    Computes the par swap rate and accrual factor.

    zcb_prices: ZCB prices in a pandas dataframe starting from p(0,0) to some maturity.
    start: First resettlement.
    maturity: Maturity of swap.
    """

    start_index = zcb_prices.columns.get_loc(start)
    maturity_index = zcb_prices.columns.get_loc(maturity)

    accrual_factors = zcb_prices.iloc[:, 1:maturity_index].sum(axis=1)
    swap_rate = (zcb_prices[start_index] - zcb_prices[maturity_index]) / accrual_factors

    return swap_rate, accrual_factors
