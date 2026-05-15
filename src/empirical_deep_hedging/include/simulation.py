import numpy as np
import pandas as pd
import QuantLib as ql
import random

from empirical_deep_hedging.include import option_functions

class Simulator():
    def __init__(self, process, periods_in_day = 1):
        self.process = process
        self.D = periods_in_day
        self.seed = None

    def set_seed(self, seed):
        self.seed = None if seed is None else int(seed)
        
    def set_properties_gbm(self, v, q, mu):
        self.v0 = v
        self.q = q
        self.mu = mu

    def set_properties_heston(self, v0, kappa, theta, sigma, rho, q, r):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.q = q
        self.r = r
            
    def simulate(self, S0, T = 252, dt = 1/252, time_grid = None, seed = None):
        if self.process == 'GBM':
            self._sim_gbm(S0, self.mu, self.v0, T, dt)
        else:
            if seed is not None:
                self.set_seed(seed)
            self._sim_heston(
                S0,
                self.v0,
                self.kappa,
                self.theta,
                self.sigma,
                self.rho,
                self.q,
                self.r,
                T,
                dt,
                time_grid = time_grid,
                seed = self.seed,
            )

    def _sim_gbm(self, S0, mu, stdev, T, dt):
        self.St = np.zeros(T)
        self.St[0] = S0
                
        for t in range(1, T):
            self.St[t] = self.St[t-1] * np.exp(mu * dt + stdev * np.sqrt(dt)*np.random.normal())

    def _sim_heston(self, S0, v0, kappa, theta, sigma, rho, q, r, T, dt, time_grid = None, seed = None):
        r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), r, ql.Actual365Fixed()))
        q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.TARGET(), q, ql.Actual365Fixed()))
        s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        process = ql.HestonProcess(r_handle, q_handle, s0_handle, v0, kappa, theta, sigma, rho)
        # An explicit time grid keeps the Heston path clock aligned with the
        # quote-date schedule used for option valuation.
        if time_grid is None:
            times = ql.TimeGrid(T / 365.0, dt)
            n_steps = int(dt)
        else:
            times = [float(x) for x in time_grid]
            n_steps = len(times) - 1
            if n_steps <= 0:
                raise ValueError("Heston time_grid must contain at least two points.")
            if abs(times[0]) > 1.0e-12:
                raise ValueError("Heston time_grid must start at 0.0.")
            if any(t1 < t0 for t0, t1 in zip(times, times[1:])):
                raise ValueError("Heston time_grid must be non-decreasing.")
        dimension = process.factors()
        if seed is None:
            uniform_rng = ql.UniformRandomGenerator()
        else:
            uniform_rng = ql.UniformRandomGenerator(int(seed))
        rng = ql.GaussianRandomSequenceGenerator(
            ql.UniformRandomSequenceGenerator(dimension * n_steps, uniform_rng)
        )
        seq = ql.GaussianMultiPathGenerator(process, list(times), rng, False)
        path = seq.next()
        values = path.value()
        St, Vt = values
        self.St = np.array([x for x in St])
        self.Vt = np.array([x for x in Vt])

    def getS(self):
        return self.St
    
    def return_set(
        self,
        strike_min,
        strike_max,
        quote_datetime,
        min_exp,
        max_exp,
        datearray,
        r,
        quote_datetimes = None,
    ):
        strike = random.uniform(strike_min, strike_max)
        strike = [strike] * len(self.St)
        
        exp = random.randint(min_exp, max_exp)
        expiration = datearray[datearray.index(quote_datetime) + int(exp)]
        expiration = [expiration] * len(self.St)
        
        if quote_datetimes is None:
            quote_datetimes = []
            
            i = 0
            while len(quote_datetimes) < len(self.St):
                temp = [datearray[datearray.index(quote_datetime) + int(i)]] * self.D
                quote_datetimes += temp
                i = i + 1
                
            quote_datetimes = quote_datetimes[:len(self.St)]
        else:
            quote_datetimes = list(quote_datetimes)
            if len(quote_datetimes) != len(self.St):
                raise ValueError(
                    "quote_datetimes length must match simulated path length: "
                    f"{len(quote_datetimes)} != {len(self.St)}"
                )

        St = self.St / self.St[0]
        
        # Store the exact pricing maturity used for synthetic option prices.
        synthetic_trading_days_elapsed = np.arange(len(self.St), dtype=float) / self.D
        synthetic_trading_days_to_expiry = exp - synthetic_trading_days_elapsed
        synthetic_tau_years = np.maximum(synthetic_trading_days_to_expiry / 252.0, 1.0e-10)

        if self.process == 'Heston':
            expiry_dt = pd.to_datetime(expiration[0])
            quote_dt = pd.to_datetime(pd.Series(quote_datetimes))
            intraday_elapsed = (np.arange(len(self.St), dtype=float) % self.D) / self.D
            heston_calendar_days_to_expiry = (
                (expiry_dt - quote_dt).dt.days.to_numpy(dtype=float) - intraday_elapsed
            )
            synthetic_tau_years = np.maximum(heston_calendar_days_to_expiry / 365.0, 1.0e-10)

        df = pd.DataFrame()
        df['underlying_bid'] = St
        df['expiration'] = expiration
        df['strike'] = strike
        df['quote_datetime'] = quote_datetimes
        df['ticker'] = 'simulated'
        if self.process in ('GBM', 'Heston'):
            df['tau_years'] = synthetic_tau_years
        
        prices = []
        
        for i in range(len(self.St)):
            if self.process == 'GBM':
                price = option_functions.call_price(St[i], strike[i], r, self.q, self.v0, synthetic_tau_years[i])
            else:
                current_v = max(self.Vt[i], 1.0e-6)

                price = option_functions.heston_price(St[i], strike[i], r, self.q, self.theta,
                    self.kappa, self.sigma, self.rho, current_v, expiration[i], quote_datetimes[i])
            prices.append(price)

        df['bid'] = prices
        df['ask'] = prices
        
        return df
