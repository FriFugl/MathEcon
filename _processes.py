from abc import ABC, abstractmethod

class StochasticProcess(ABC):
    """Represente a Stochastic process"""

    @abstractmethod
    def simulate(self): ...