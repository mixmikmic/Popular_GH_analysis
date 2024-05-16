import dx
import datetime as dt

# constant short rate
r = dx.constant_short_rate('r', 0.01)

# geometric Brownian motion
me = dx.market_environment('me', dt.datetime(2015, 1, 1))
me.add_constant('initial_value', 100.)
me.add_constant('volatility', 0.2)
me.add_constant('final_date', dt.datetime(2015, 12, 31))
me.add_constant('currency', 'EUR')

# jump component
me.add_constant('lambda', 0.4)
me.add_constant('mu', -0.6)
me.add_constant('delta', 0.2)

# stochastic volatiltiy component
me.add_constant('rho', -.5)
me.add_constant('kappa', 5.0)
me.add_constant('theta', 0.02)
me.add_constant('vol_vol', 0.3)

# valuation environment
val_env = dx.market_environment('val_env', dt.datetime(2015, 1, 1))
val_env.add_constant('paths', 55000)
    # 25,000 paths
val_env.add_constant('frequency', 'D')
    # weekly frequency
val_env.add_curve('discount_curve', r)
val_env.add_constant('starting_date', dt.datetime(2015, 1, 1))
val_env.add_constant('final_date', dt.datetime(2015, 12, 31))

# add valuation environment to market environment
me.add_environment(val_env)

gbm = dx.geometric_brownian_motion('gbm', me)

jd = dx.jump_diffusion('jd', me)

sv = dx.stochastic_volatility('sv', me)

svjd = dx.stoch_vol_jump_diffusion('svjd', me)

# market environment for the options
me_option = dx.market_environment('option', dt.datetime(2015, 1, 1))
me_option.add_constant('maturity', dt.datetime(2015, 12, 31))
me_option.add_constant('strike', 100.)
me_option.add_constant('currency', 'EUR')
me_option.add_environment(me)
me_option.add_environment(val_env)

euro_put_gbm = dx.valuation_mcs_european_single('euro_put', gbm, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_gbm = dx.valuation_mcs_european_single('euro_call', gbm, me_option,
                                  'np.maximum(maturity_value - strike, 0)')

euro_put_jd = dx.valuation_mcs_european_single('euro_put', jd, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_jd = dx.valuation_mcs_european_single('euro_call', jd, me_option,
                                  'np.maximum(maturity_value - strike, 0)')

euro_put_sv = dx.valuation_mcs_european_single('euro_put', sv, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_sv = dx.valuation_mcs_european_single('euro_call', sv, me_option,
                                  'np.maximum(maturity_value - strike, 0)')

euro_put_svjd = dx.valuation_mcs_european_single('euro_put', svjd, me_option,
                                  'np.maximum(strike - maturity_value, 0)')
euro_call_svjd = dx.valuation_mcs_european_single('euro_call', svjd, me_option,
                                  'np.maximum(maturity_value - strike, 0)')

import numpy as np
import pandas as pd

freq = '2m'  # used for maturity definitions
periods = 3  # number of intervals for maturity grid
strikes = 5  # number of strikes per maturity
initial_value = 100  # initial value for all risk factors
start = 0.8  # lowest strike in percent of spot
end = 1.2  # highest strike in percent of spot
start_date = '2015/3/1'  # start date for simulation/pricing

euro_put_gbm.present_value()
  # method call needed for initialization

bsm_option = dx.BSM_european_option('bsm_opt', me_option)

get_ipython().run_cell_magic('time', '', "# European put\nprint('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))\nfor maturity in pd.date_range(start=start_date, freq=freq, periods=periods):\n    bsm_option.maturity = maturity\n    euro_put_gbm.update(maturity=maturity)\n    for strike in np.linspace(start, end, strikes) * initial_value:\n        T = (maturity - me_option.pricing_date).days / 365.\n        euro_put_gbm.update(strike=strike)\n        mcs = euro_put_gbm.present_value()\n        bsm_option.strike = strike\n        ana = bsm_option.put_value()\n        print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.2f '\n                % (T, strike, mcs, ana, mcs - ana, (mcs - ana) / ana * 100))")

euro_call_gbm.present_value()
  # method call needed for initialization

get_ipython().run_cell_magic('time', '', "# European calls\nprint('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))\nfor maturity in pd.date_range(start=start_date, freq=freq, periods=periods):\n    euro_call_gbm.update(maturity=maturity)\n    for strike in np.linspace(start, end, strikes) * initial_value:\n        T = (maturity - me_option.pricing_date).days / 365.\n        euro_call_gbm.update(strike=strike)\n        mcs = euro_call_gbm.present_value()\n        bsm_option.strike = strike\n        bsm_option.maturity = maturity\n        ana = bsm_option.call_value()\n        print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.2f ' \\\n                % (T, strike, mcs, ana, mcs - ana, (mcs - ana) / ana * 100))")

def valuation_benchmarking(valuation_object, fourier_function):
    print('%4s  | %7s | %7s | %7s | %7s | %7s' % ('T', 'strike', 'mcs', 'fou', 'dif', 'rel'))
    for maturity in pd.date_range(start=start_date, freq=freq, periods=periods):
        valuation_object.update(maturity=maturity)
        me_option.add_constant('maturity', maturity)
        for strike in np.linspace(start, end, strikes) * initial_value:
            T = (maturity - me_option.pricing_date).days / 365.
            valuation_object.update(strike=strike)
            mcs = valuation_object.present_value()
            me_option.add_constant('strike', strike)
            fou = fourier_function(me_option)
            print('%4.3f | %7.3f | %7.4f | %7.4f | %7.4f | %7.3f '
                % (T, strike, mcs, fou, mcs - fou, (mcs - fou) / fou * 100))

euro_put_jd.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_put_jd, dx.M76_put_value)')

euro_call_jd.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_call_jd, dx.M76_call_value)')

euro_put_sv.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_put_sv, dx.H93_put_value)')

euro_call_sv.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_call_sv, dx.H93_call_value)')

euro_put_svjd.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_put_svjd, dx.B96_put_value)')

euro_call_svjd.present_value()
  # method call needed for initialization

get_ipython().magic('time valuation_benchmarking(euro_call_svjd, dx.B96_call_value)')

