import numpy as np
from scipy.stats import norm

class BSPath:

    def __init__(
            self,
            r: float,          # risk-free interest rate
            sigma: float,      # volatility of the underlying
            S0: float          # spot at time t=0 
    ) -> None:
        self.rate = r
        self.vol = sigma
        self.spot = S0

    def simulation(self, m_time_steps, length_time_steps, n_simulations):
        dt = length_time_steps
        m, n = m_time_steps, n_simulations
        dw = np.sqrt(dt)*self.vol*np.random.randn(m, n)  # brownian motion increments x sigma (independent)        
        wt = np.cumsum(dw, axis=0)                       # brownian motion values for time t = 1, ..., m
        drift = (self.rate - self.vol**2/2)*dt*np.arange(1, m + 1).reshape((m, 1))
        s = self.spot*np.exp(drift + wt)                 # trajectory at time t = 1, ..., m
        return s
