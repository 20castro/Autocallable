import numpy as np
from scipy.stats import norm
import pandas as pd

class MonteCarlo:

    def __init__(
            self,
            rate,
            q,
            barrier,
            freq,
            mat,
            put_strike,
            notional
    ):
        self.coupon = q
        self.rate = rate
        self.barrier = barrier
        self.strike = put_strike
        self.notional = notional
        self.frequency = freq
        self.maturity = mat

    def _barrier_breach_time(self, simulation: np.ndarray):
        breach = np.max(simulation, axis=0) >= self.barrier
        value_at_expiry = simulation[-1, np.logical_not(breach)]
        breach_sim = simulation[::, breach]
        m, nb = breach_sim.shape
        first_breach = np.zeros(nb)
        for k in range(m, 0, -1):
            first_breach = np.where(breach_sim[k - 1, ::] >= self.barrier, k, first_breach)
        assert np.all(first_breach >= 1)
        return first_breach, value_at_expiry
    
    def valuation(self, simulation: np.ndarray, verbose=True):
        first_breach, value_at_expiry = self._barrier_breach_time(simulation)
        n_breach = len(first_breach)
        n_expire = len(value_at_expiry)
        n_put_exercised = np.count_nonzero(value_at_expiry <= self.strike)
        n = n_breach + n_expire
        if verbose:
            print(f'{(100*n_breach/n):.2f}% hit the barrier and the put is exercised {(100*n_put_exercised/n):.2f}% of the time')
            first_breach_count = pd.value_counts(first_breach)/n
            print('Barrier breach at time:')
            for t, cnt in first_breach_count.sort_index().items():
                print(f'- {t*self.frequency} years: {100*cnt:.2f}% of the time')
        discount_autocall = np.exp(-self.rate*first_breach*self.frequency)
        payoff_in_case_of_breach = discount_autocall*self.notional*(1 + first_breach*self.coupon)
        discount_maturity = np.exp(-self.rate*self.maturity)
        put_payoff = np.maximum(0, self.strike - value_at_expiry)
        payoff_without_breach = discount_maturity*(self.notional - put_payoff)
        payoff = np.concatenate([payoff_in_case_of_breach, payoff_without_breach])
        est = np.mean(payoff)
        ci95 = 1.96*np.std(payoff)/np.sqrt(n)
        return est, ci95
        