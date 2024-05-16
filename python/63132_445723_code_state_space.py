import numpy as np

def kalman_filter(y, Z, H, T, Q, a_0, P_0):
    # Dimensions
    k_endog, nobs = y.shape
    k_states = T.shape[0]

    # Allocate memory for variables
    filtered_state = np.zeros((k_states, nobs))
    filtered_state_cov = np.zeros((k_states, k_states, nobs))
    predicted_state = np.zeros((k_states, nobs+1))
    predicted_state_cov = np.zeros((k_states, k_states, nobs+1))
    forecast = np.zeros((k_endog, nobs))
    forecast_error = np.zeros((k_endog, nobs))
    forecast_error_cov = np.zeros((k_endog, k_endog, nobs))
    loglikelihood = np.zeros((nobs+1,))

    # Copy initial values to predicted
    predicted_state[:, 0] = a_0
    predicted_state_cov[:, :, 0] = P_0

    # Kalman filter iterations
    for t in range(nobs):

        # Forecast for time t
        forecast[:, t] = np.dot(Z, predicted_state[:, t])

        # Forecast error for time t
        forecast_error[:, t] = y[:, t] - forecast[:, t]

        # Forecast error covariance matrix and inverse for time t
        tmp1 = np.dot(predicted_state_cov[:, :, t], Z.T)
        forecast_error_cov[:, :, t] = (
            np.dot(Z, tmp1) + H
        )
        forecast_error_cov_inv = np.linalg.inv(forecast_error_cov[:, :, t])
        determinant = np.linalg.det(forecast_error_cov[:, :, t])

        # Filtered state for time t
        tmp2 = np.dot(forecast_error_cov_inv, forecast_error[:,t])
        filtered_state[:, t] = (
            predicted_state[:, t] +
            np.dot(tmp1, tmp2)
        )

        # Filtered state covariance for time t
        tmp3 = np.dot(forecast_error_cov_inv, Z)
        filtered_state_cov[:, :, t] = (
            predicted_state_cov[:, :, t] -
            np.dot(
                np.dot(tmp1, tmp3),
                predicted_state_cov[:, :, t]
            )
        )

        # Loglikelihood
        loglikelihood[t] = -0.5 * (
            np.log((2*np.pi)**k_endog * determinant) +
            np.dot(forecast_error[:, t], tmp2)
        )

        # Predicted state for time t+1
        predicted_state[:, t+1] = np.dot(T, filtered_state[:, t])

        # Predicted state covariance matrix for time t+1
        tmp4 = np.dot(T, filtered_state_cov[:, :, t])
        predicted_state_cov[:, :, t+1] = np.dot(tmp4, T.T) + Q
        
        predicted_state_cov[:, :, t+1] = (
            predicted_state_cov[:, :, t+1] + predicted_state_cov[:, :, t+1].T
        ) / 2

    return (
        filtered_state, filtered_state_cov,
        predicted_state, predicted_state_cov,
        forecast, forecast_error, forecast_error_cov,
        loglikelihood
    )

from scipy.signal import lfilter

# Parameters
nobs = 100
phi = 0.5
sigma2 = 1.0

# Example dataset
np.random.seed(1234)
eps = np.random.normal(scale=sigma2**0.5, size=nobs)
y = lfilter([1], [1, -phi], eps)[np.newaxis, :]

# State space
Z = np.array([1.])
H = np.array([0.])
T = np.array([phi])
Q = np.array([sigma2])

# Initial state distribution
a_0 = np.array([0.])
P_0 = np.array([sigma2 / (1 - phi**2)])

# Run the Kalman filter
res = kalman_filter(y, Z, H, T, Q, a_0, P_0)

import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm

# True model parameters for an AR(1)
nobs = int(1e3)
true_phi = 0.5
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model for an ARMA(1,1)
class ARMA11(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(ARMA11, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1., 0]
        self['transition'] = [[0, 0],
                              [1., 0]]
        self['selection', 0, 0] = 1.

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(ARMA11, self).update(params, transformed, **kwargs)

        self['design', 0, 1] = params[0]
        self['transition', 0, 0] = params[1]
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.,0.,1]  # these are very simple

# Create and fit the model
mod = ARMA11(endog)
res = mod.fit()
print(res.summary())

import statsmodels.api as sm
from pandas_datareader.data import DataReader

gdp = DataReader('GDPC1', 'fred', start='1959', end='12-31-2014')

# Create the model, here an SARIMA(1,1,1) x (0,1,1,4) model
mod = sm.tsa.SARIMAX(gdp, order=(1,1,1), seasonal_order=(0,1,1,4))

# Fit the model via maximum likelihood
res = mod.fit()
print(res.summary())

"""
Univariate Local Linear Trend Model
"""
import pandas as pd

class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, trend=True):
        # Model properties
        self.trend = trend

        # Model order
        k_states = 2
        k_posdef = 1 + self.trend

        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
        )

        # Initialize the matrices
        self['design'] = np.array([1, 0])
        self['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self['selection'] = np.eye(k_states)[:,:k_posdef]

        # Initialize the state space model as approximately diffuse
        self.initialize_approximate_diffuse()
        # Because of the diffuse initialization, burn first two loglikelihoods
        self.loglikelihood_burn = 2

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

        # The parameters depend on whether or not we have a trend
        param_names = ['sigma2.measurement', 'sigma2.level']
        if self.trend:
            param_names += ['sigma2.trend']
        self._param_names = param_names

    @property
    def start_params(self):
        return [0.1] * (2 + self.trend)

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self['obs_cov',0,0] = params[0]

        # State covariance
        self[self._state_cov_idx] = params[1:]

y = sm.datasets.nile.load_pandas().data
y.index = pd.date_range('1871', '1970', freq='AS')

mod1 = LocalLinearTrend(y['volume'], trend=True)
res1 = mod1.fit()
print res1.summary()

mod2 = LocalLinearTrend(y['volume'], trend=False)
res2 = mod2.fit()
print res2.summary()

