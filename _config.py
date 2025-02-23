import numpy as np

polynomial_classes = {
    "power": np.polynomial.polynomial.Polynomial,
    "chebyshev": np.polynomial.chebyshev.Chebyshev,
    "legendre": np.polynomial.legendre.Legendre,
    "laguerre": np.polynomial.laguerre.Laguerre,
    "hermite": np.polynomial.hermite.Hermite,
}
