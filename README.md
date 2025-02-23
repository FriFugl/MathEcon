\section*{Least squares monte carlo method for American option pricing}
This is an implementation of the least-squares monte carlo method developed by Longstaff & Schwartz to price american style options. It has been implemented to be capable of pricing Bermudan swaptions and American (or Bermudan) stock options.

The _processes.py file contains a class representing a stochastic process.
The _short_rate_models.py file contains classes of short rate models to simulate short rate trajectories and calculate zero coupon bond prices.
The _stock_path_models.py file contains a class for the Geometric Brownian Motion process.
The _LSM.py contains a classes for different Least-Squares Monte-Carlo methods for for option pricing.


