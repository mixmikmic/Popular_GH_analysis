get_ipython().magic('pylab inline')
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

npts=100
X = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],npts)
X = X-np.mean(X,0)  

params  = [1,2]
y_noise = 0.2
Y = np.dot(X,params) + y_noise*np.random.randn(npts)
Y = Y-np.mean(Y)    # remove mean so we can skip ones in design mtx

for i in range(2):
    print('correlation(X[%d],Y))'%i, '= %4.3f' % np.corrcoef(X[:,i],Y)[0,1])
    plt.subplot(1,2,i+1)
    plt.scatter(X[:,i],Y)

params_est =  np.linalg.lstsq(X,Y)[0]

print(params_est)

x0_slope=numpy.linalg.lstsq(X[:,0].reshape((npts,1)),X[:,1].reshape((npts,1)))[0]

X_orth=X.copy()

X_orth[:,1]=X[:,1] - X[:,0]*x0_slope
print('Correlation matrix for original design matrix')
print (numpy.corrcoef(X.T))

print ('Correlation matrix for orthogonalized design matrix')
print (numpy.corrcoef(X_orth.T))

params_est_orth =  numpy.linalg.lstsq(X_orth,Y)[0]

print (params_est_orth)

# Make X nptsx10
X = np.random.normal(0,1,(npts,10))
X = X - X.mean(axis=0)
X0 = X[:,:2]
X1 = X[:,2:]

# Orthogonolizing X0 with respect to X1: 
X0_orthog_wrt_X1 = X0 - np.dot(X1,np.linalg.pinv(X1)).dot(X0)

# reconstruct the new X matrix : Xorth
Xorth = np.hstack((X0_orthog_wrt_X1, X1))

# checking that the covariance of the two first regressors with others is 0
# look at the 5 first regressors
print (np.corrcoef(Xorth.T)[:5,:5])



