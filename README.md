# Least Squares Monte Marlo methods for American option pricing
This is an implementation of the least-squares monte carlo method developed by Longstaff & Schwartz to price american style options. It has been implemented to be capable of pricing Bermudan swaptions and American (or Bermudan) stock options.

* `_processes` file contains a class representing a stochastic process.
* `_short_rate_models` file contains classes of short rate models to simulate short rate trajectories and calculate zero coupon bond prices.
* `_stock_path_models` file contains a class for the Geometric Brownian Motion process.
* `_LSM` contains a classes for different Least Squares Monte Carlo methods for for option pricing.


