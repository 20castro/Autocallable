import numpy as np
from scipy.stats import norm
from pandas import value_counts

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
        first_breach = self.frequency*(1 + np.argmax(simulation[::, breach], axis=0))
        return first_breach, value_at_expiry
    
    def valuation(self, simulation: np.ndarray, verbose=True):
        first_breach, value_at_expiry =  self._barrier_breach_time(simulation)
        n_breach = len(first_breach)
        n_expire = len(value_at_expiry)
        n_put_exercised = np.count_nonzero(value_at_expiry <= self.strike)
        n = n_breach + n_expire
        if verbose:
            print(f'{(100*n_breach/n):.2f}% hit the barrier and the put is exercised {(100*n_put_exercised/n):.2f}% of the time')
        discount_autocall = np.exp(-self.rate*first_breach)
        payoff_in_case_of_breach = discount_autocall*self.notional*(1 + first_breach*self.coupon)
        discount_maturity = np.exp(-self.rate*self.maturity)
        put_payoff = np.maximum(0, self.strike - value_at_expiry)
        payoff_without_breach = discount_maturity*(self.notional - put_payoff)
        payoff = np.concatenate([payoff_in_case_of_breach, payoff_without_breach])
        est = np.mean(payoff)
        ci95 = 1.96*np.std(payoff)/np.sqrt(n)
        return est, ci95
        