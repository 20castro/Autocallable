import numpy as np

class HestonPath:

    def __init__(
            self,
            r: float,          # risk-free interest rate
            S0: float,         # spot at time t=0
            V0: float,         # initial variance
            kappa: float,
            theta: float,
            rho: float,
            sigmav: float      # volatility of the volatility
    ):
        self.rate = r
        self.S0 = S0
        self.V0 = V0
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.sigmav = sigmav

    def _simulate_correlated_BM(self, m_time_steps, length_time_steps, n_simulations):
        sdt = np.sqrt(length_time_steps)
        db1 = sdt*np.random.randn(m_time_steps, n_simulations)
        db2 = sdt*np.random.randn(m_time_steps, n_simulations)
        dw2 = self.rho*db1 + np.sqrt(1 - self.rho**2)*db2
        return db1, dw2

    def simulation(self, m_time_steps, length_time_steps, n_simulations):
        dt = length_time_steps
        m, n = m_time_steps, n_simulations
        dw1, dw2 = self._simulate_correlated_BM(m, dt, n)
        S = self.S0*np.ones(shape=(m + 1, n))
        V = self.V0*np.ones(shape=(m + 1, n))
        for k, (dwk1, dwk2) in enumerate(zip(dw1, dw2)):
            St = S[k, :]
            Vt =  np.maximum(0, V[k, :])
            V[k + 1, :] = Vt + self.kappa*(self.theta - Vt)*dt + self.sigmav*np.sqrt(Vt)*dwk2
            S[k + 1, :] = St*(1 + self.rate*dt + np.sqrt(Vt)*dwk1)
        return S
    
    def discrete_observations(self, m_observations, m_by_period, duration_period, n_simulations):
        dt = duration_period/m_by_period
        m, n = m_observations*m_by_period, n_simulations
        S = self.simulation(m, dt, n)
        return S[m_by_period::m_by_period, ::]
