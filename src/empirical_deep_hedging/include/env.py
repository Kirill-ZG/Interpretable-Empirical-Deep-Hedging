import numpy as np
import pandas as pd
import os

from datetime import datetime
from scipy.stats import norm
from empirical_deep_hedging.include.option_functions import calc_impl_volatility
from empirical_deep_hedging.include import option_functions
from empirical_deep_hedging.include import data_keeper
from empirical_deep_hedging.include import simulation
from empirical_deep_hedging.include.settings import getSettings

class Env():
    def __init__(self, s = getSettings()):
        self.sim = simulation.Simulator(s['process'], periods_in_day = s['D'])
        
        self.transaction_cost = s['transaction_cost']
        self.kappa = s['kappa']
        self.reward_exponent = s['reward_exponent']
        self.SIGMA = s['SIGMA']
        self.process = s['process']

        if self.process == 'GBM':
            self.days_in_year = 252.0
        else:
            self.days_in_year = 365.0

        self.q = s['q']
        data_dir = os.environ.get('DATA_DIR', 'data')
        self.r_df = pd.read_csv(os.path.join(data_dir, '1yr_treasury.csv'))
        heston_params_path = s.get(
            'heston_params_path',
            os.path.join(data_dir, 'heston_params.csv')
        )
        if not os.path.exists(heston_params_path):
            heston_params_path = 'data/heston_params.csv'
        self.heston_params = pd.read_csv(heston_params_path)

        self.D, self.steps = s['D'], s['n_steps']
        if self.process == 'Real':
            self.data_keeper = data_keeper.DataKeeper(self.steps)
        self.data_set = pd.DataFrame()
        self.t, self.v, self.date_idx, = 0, 0.0, 0
        
        self.option = {}
        self.S = []
        
    def get_bs_delta(self):
        # The benchmark hedge is computed from spot moneyness. The policy state
        # uses forward moneyness, so using it here would count carry twice.
        tau = self.option['T'] / self.days_in_year
        d1, _ = option_functions._d(self.S[self.t], self.K, self.r, self.q, self.v, tau)
        return np.exp(-self.q * tau) * norm.cdf(d1)

    def __concat_state(self):
        return np.array([self.option['S/K'], self.option['T']/30, self.stockOwned, self.v])
    
    def __update_option(self):
        row = self.data_set.loc[self.t, :]

        spot = row['underlying_bid']
        P = 0.5 * (row['bid'] + row['ask'])
        self.expiry = row['expiration'][0:10]
        self.K = float(row['strike'])
        self.S[self.t] = spot
        self.cur_date = row['quote_datetime'][0:10]
        self.ticker = row['ticker']
        self.option['P'] = P

        try:
            self.r = self.r_df.loc[self.r_df['Date'] == self.cur_date, '1y'].iloc[0]
        except:
            print(f"R MISSING on {self.cur_date}! Carrying over r={self.r}")
        
        # Synthetic rows carry their pricing maturity directly; empirical rows
        # reconstruct maturity from quote and expiration dates.
        if 'tau_years' in row.index and pd.notna(row['tau_years']):
            tau = float(row['tau_years'])
            ttm = tau * self.days_in_year
        else:
            ttm = (datetime.strptime(self.expiry, '%Y-%m-%d') - \
                datetime.strptime(self.cur_date, '%Y-%m-%d')).days - (1 - (self.D - self.t%self.D) / self.D)
            tau = ttm / self.days_in_year

        self.option['T'] = ttm
        self.option['tau_years'] = tau
        # The network observes forward moneyness while diagnostics retain both
        # spot and forward conventions.
        forward_spot = spot * np.exp((self.r - self.q) * tau)
        self.option['spot_S/K'] = spot / self.K
        self.option['forward_S/K'] = forward_spot / self.K
        self.option['S/K'] = self.option['forward_S/K']
              
        if self.process == 'GBM':
            self.v = self.SIGMA
        else:
            iv = calc_impl_volatility(spot, self.K, self.r, self.q, tau, P)
            if iv:
                self.v = iv
            else:
                print(f"IV SOLVER FAILED! Date: {self.cur_date}, Strike: {self.K}, Carryover Vol: {self.v:.4f}")
        
    def reset(self, testing = False, start_a = 0.0, start_b = 0.0):
        self.testing = testing
        self.t = 0
        self.S = np.zeros(self.steps + 1)
        
        self.stockOwned, self.b_stockOwned = start_a, start_b
        
        new_set = None
        
        if testing:
            self.data_set = self.data_keeper.next_test_set()
        else:
            if self.process == 'Real':
                self.data_set = self.data_keeper.next_train_set()
            else:
                while new_set is None:
                    datearray = sorted(self.r_df['Date'].astype(str).unique())
                    max_future_index = 90

                    if self.process == 'Heston':
                        heston_dates = set(self.heston_params['date'].astype(str))
                        date_positions = {d: i for i, d in enumerate(datearray)}
                        dates = [
                            d for d in datearray[:-max_future_index]
                            if d >= '2013-01-01' and d in heston_dates
                        ]
                        dates = [
                            d for d in dates
                            if date_positions[d] + max_future_index < len(datearray)
                        ]
                    else:
                        dates = [d for d in datearray if d >= '2013-01-01']
                        dates = dates[:-max_future_index]

                    if not dates:
                        raise ValueError(
                            f"No valid synthetic start dates found for process={self.process}."
                        )
                    quote_datetime = np.random.choice(dates)

                    try:
                        self.r = self.r_df.loc[self.r_df['Date'] == quote_datetime, '1y'].iloc[0]
                    except:
                        self.r = 0.01
                    
                    if self.process == 'GBM':
                        self.sim.set_properties_gbm(self.SIGMA, self.q, .0)
                        T = self.steps + 1
                        dt = 1 / (self.days_in_year * self.D)
                    else:
                        params = self.heston_params[self.heston_params['date'] == quote_datetime]
                        if params.empty:
                            continue

                        v0 = params.iloc[0]['v0']
                        kappa = params.iloc[0]['kappa']
                        theta = params.iloc[0]['theta']
                        sigma = params.iloc[0]['sigma']
                        rho = params.iloc[0]['rho']

                        self.sim.set_properties_heston(v0, kappa, theta, sigma, rho, self.q, self.r)

                        # Use the same calendar clock for path simulation,
                        # option valuation, and quote-date labels.
                        start_idx = datearray.index(quote_datetime)
                        start_dt = datetime.strptime(quote_datetime, '%Y-%m-%d')
                        quote_datetimes = []
                        heston_time_grid = []
                        for step_i in range(self.steps + 1):
                            date_idx = start_idx + step_i // self.D
                            if date_idx >= len(datearray):
                                quote_datetimes = []
                                break
                            cur_date = datearray[date_idx]
                            cur_dt = datetime.strptime(cur_date, '%Y-%m-%d')
                            intraday_elapsed = (step_i % self.D) / self.D
                            elapsed_days = (cur_dt - start_dt).days + intraday_elapsed
                            quote_datetimes.append(cur_date)
                            heston_time_grid.append(elapsed_days / 365.0)

                        if not quote_datetimes:
                            continue

                        ql_seed = int(np.random.randint(1, 2**31 - 1))
                        self.sim.simulate(
                            1.0,
                            self.steps,
                            self.steps,
                            time_grid = heston_time_grid,
                            seed = ql_seed,
                        )
                        new_set = self.sim.return_set(
                            .85,
                            1.15,
                            quote_datetime,
                            30,
                            90,
                            datearray,
                            self.r,
                            quote_datetimes = quote_datetimes,
                        )
                        self.data_set = new_set
                        continue

                    self.sim.simulate(1.0, T, dt)
                    new_set = self.sim.return_set(
                        .85,
                        1.15,
                        quote_datetime,
                        30,
                        90,
                        datearray,
                        self.r,
                    )
                    self.data_set = new_set
            
        self.__update_option()
        return self.__concat_state()

    def step(self, delta):
        def reward_func(pnl):
            pnl *= 100
            reward = 0.03 + pnl - self.kappa * (abs(pnl)**self.reward_exponent)
            return reward * 10
        
        infos = {'T':self.option['T'],
                'S/K':self.option['S/K']}
        infos['spot S/K'] = self.option.get('spot_S/K', np.nan)
        infos['forward S/K'] = self.option.get('forward_S/K', np.nan)
        infos['TauYears'] = self.option.get('tau_years', self.option['T'] / self.days_in_year)
        
        infos['Date'] = self.cur_date
        start_r = self.r
        start_q = self.q
        infos['DateStep'] = self.t % self.D
        
        b_delta = self.get_bs_delta()
        
        # Linear transaction cost based on current face value and position change.
        t_cost = -abs(-delta - self.stockOwned) * self.S[self.t] * self.transaction_cost
        b_t_cost =  -abs(-b_delta - self.b_stockOwned) * self.S[self.t] * self.transaction_cost
        
        opt_old_price = self.option['P']
        
        self.t += 1
        
        self.__update_option()
        infos['DateEnd'] = self.cur_date
        infos['r'] = start_r
        infos['q'] = start_q
        
        done = self.t >= self.steps

        opt_new_price = self.option['P']

        pnl = -delta * (self.S[self.t] - self.S[self.t - 1])
        b_pnl = -b_delta * (self.S[self.t] - self.S[self.t - 1])
        
        pnl += (opt_new_price - opt_old_price) + t_cost
        b_pnl += (opt_new_price - opt_old_price) + b_t_cost        
        
        self.stockOwned = -delta
        self.b_stockOwned = -b_delta

        reward = reward_func(pnl)
        b_reward = reward_func(b_pnl)
        
        infos['B Reward'] = b_reward
        infos['A Reward'] = reward
        infos['A PnL'] = pnl
        infos['B PnL'] = b_pnl      
        infos['P0'] = opt_new_price  
        infos['P-1'] = opt_old_price
        infos['S0'] = self.S[self.t]
        infos['S-1'] = self.S[self.t - 1]
        infos['A Pos'] = self.stockOwned
        infos['B Pos'] = self.b_stockOwned
        infos['A TC'] = t_cost
        infos['B TC'] = b_t_cost
        infos['A PnL - TC'] = pnl - t_cost
        infos['B PnL - TC'] = b_pnl - b_t_cost
        infos['Expiry'] = self.expiry
        infos['v'] = self.v
        
        return self.__concat_state(), reward, done, infos
