import numpy,pandas
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.preprocessing import PolynomialFeatures,scale
from sklearn.linear_model import LinearRegression,LassoCV,Ridge
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.tools.tools import add_constant

recreate=True
if recreate:
    seed=20698
else:
    seed=numpy.ceil(numpy.random.rand()*100000).astype('int')
    print(seed)

numpy.random.seed(seed)

def make_continuous_data(mean=[45,100],var=[10,10],cor=-0.6,N=100):
    """
    generate a synthetic data set with two variables
    """
    cor=numpy.array([[1.,cor],[cor,1.]])
    var=numpy.array([[var[0],0],[0,var[1]]])
    cov=var.dot(cor).dot(var)
    return numpy.random.multivariate_normal(mean,cov,N)

n=50
d=make_continuous_data(N=n)
y=d[:,1]
plt.scatter(d[:,0],d[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')

print('data R-squared: %f'%numpy.corrcoef(d.T)[0,1]**2)

def loglike(y,yhat,s2=None,verbose=True):
    N = len(y)
    SSR = numpy.sum((y-yhat)**2)
    if s2 is None:
        # use observed stdev
        s2 = SSR / float(N)
    logLike = -(n/2.)*numpy.log(s2) - (n/2.)*numpy.log(2*numpy.pi) - SSR/(2*s2)
    if verbose:
        print('SSR:',SSR)
        print('s2:',s2)
        print('logLike:',logLike)
    return logLike
    

logLike_null=loglike(y,numpy.zeros(len(y)),s2=1)

mean=numpy.mean(y)
print('mean:',mean)
pred=numpy.ones(len(y))*mean
logLike_mean=loglike(y,pred,s2=1)

var=numpy.var(y)
print('variance',var)
pred=numpy.ones(len(y))*mean
logLike_mean_std=loglike(y,pred)

X=d[:,0]
X=add_constant(X)
result = sm.OLS( y, X ).fit()
print(result.summary())
intercept=result.params[0]
slope=result.params[1]
pred=result.predict(X)
logLike_ols=loglike(y,pred)
plt.scatter(y,pred)

print('processing speed = %f + %f*age'%(intercept,slope))
print('p =%f'%result.pvalues[1])

def get_RMSE(y,pred):
    return numpy.sqrt(numpy.mean((y - pred)**2))

def get_R2(y,pred):
    """ compute r-squared"""
    return numpy.corrcoef(y,pred)[0,1]**2

ax=plt.scatter(d[:,0],d[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')
plt.plot(d[:,0], slope * d[:,0] + intercept, color='red')
# plot residual lines
d_predicted=slope*d[:,0] + intercept
for i in range(d.shape[0]):
    x=d[i,0]
    y=d[i,1]
    plt.plot([x,x],[d_predicted[i],y],color='blue')

RMSE=get_RMSE(d[:,1],d_predicted)
rsq=get_R2(d[:,1],d_predicted)
print('rsquared=%f'%rsq)

d_new=make_continuous_data(N=n)
d_new_predicted=intercept + slope*d_new[:,0]
RMSE_new=get_RMSE(d_new[:,1],d_new_predicted)
rsq_new=get_R2(d_new[:,1],d_new_predicted)
print('R2 for new data: %f'%rsq_new)

ax=plt.scatter(d_new[:,0],d_new[:,1])
plt.xlabel('age')
plt.ylabel('processing speed')
plt.plot(d_new[:,0], slope * d_new[:,0] + intercept, color='red')

nruns=100
slopes=numpy.zeros(nruns)
intercepts=numpy.zeros(nruns)
rsquared=numpy.zeros(nruns)

fig = plt.figure()
ax = fig.gca()

for i in range(nruns):
    data=make_continuous_data(N=n)
    slopes[i],intercepts[i],_,_,_=scipy.stats.linregress(data[:,0],data[:,1])
    ax.plot(data[:,0], slopes[i] * data[:,0] + intercepts[i], color='red', alpha=0.05)
    pred_orig=intercept + slope*data[:,0]
    rsquared[i]=get_R2(data[:,1],pred_orig)

print('Original R2: %f'%rsq)
print('Mean R2 for new datasets on original model: %f'%numpy.mean(rsquared))

# initialize the sklearn leave-one-out operator
loo=LeaveOneOut()  

for train,test in loo.split(range(10)):
    print('train:',train,'test:',test)

# initialize the sklearn leave-one-out operator
kf=KFold(n_splits=5,shuffle=True)  

for train,test in kf.split(range(10)):
    print('train:',train,'test:',test)

loo=LeaveOneOut()

slopes_loo=numpy.zeros(n)
intercepts_loo=numpy.zeros(n)
pred=numpy.zeros(n)

ctr=0
for train,test in loo.split(range(n)):
    slopes_loo[ctr],intercepts_loo[ctr],_,_,_=scipy.stats.linregress(d[train,0],d[train,1])
    pred[ctr]=intercepts_loo[ctr] + slopes_loo[ctr]*data[test,0]
    ctr+=1

print('R2 for leave-one-out prediction: %f'%get_R2(pred,data[:,1]))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
_=plt.hist(slopes_loo,20)
plt.xlabel('slope estimate')
plt.ylabel('frequency')
plt.subplot(1,2,2)
_=plt.hist(intercepts_loo,20)
plt.xlabel('intercept estimate')
plt.ylabel('frequency')

# add an outlier
data_null=make_continuous_data(N=n,cor=0.0)
outlier_multiplier=2.0

data=numpy.vstack((data_null,[numpy.max(data_null[:,0])*outlier_multiplier,
                         numpy.max(data_null[:,1])*outlier_multiplier*-1]))
plt.scatter(data[:,0],data[:,1])
slope,intercept,r,p,se=scipy.stats.linregress(data[:,0],data[:,1])
plt.plot([numpy.min(data[:,0]),intercept + slope*numpy.min(data[:,0])],
         [numpy.max(data[:,0]),intercept + slope*numpy.max(data[:,0])])
rsq_outlier=r**2
print('R2 for regression with outlier: %f'%rsq_outlier)

loo=LeaveOneOut()

pred_outlier=numpy.zeros(data.shape[0])

ctr=0
for train,test in loo.split(range(data.shape[0])):
    s,i,_,_,_=scipy.stats.linregress(data[train,0],data[train,1])
    pred_outlier[ctr]=i + s*data[test,0]
    ctr+=1

print('R2 for leave-one-out prediction: %f'%get_R2(pred_outlier,data[:,1]))

# from https://gist.github.com/iizukak/1287876
def gram_schmidt_columns(X):
    Q, R = numpy.linalg.qr(X)
    return Q

def make_continuous_data_poly(mean=0,var=1,betaval=5,order=1,N=100):
    """
    generate a synthetic data set with two variables
    allowing polynomial functions up to 5-th order
    """
    x=numpy.random.randn(N)
    x=x-numpy.mean(x)
    pf=PolynomialFeatures(5,include_bias=False)

    x_poly=gram_schmidt_columns(pf.fit_transform(x[:,numpy.newaxis]))

    betas=numpy.zeros(5)
    betas[0]=mean
    for i in range(order):
        betas[i]=betaval
    func=x_poly.dot(betas)+numpy.random.randn(N)*var
    d=numpy.vstack((x,func)).T
    return d,x_poly

n=25
trueorder=2
data,x_poly=make_continuous_data_poly(N=n,order=trueorder)

# fit models of increasing complexity
npolyorders=7

plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.title('fitted data')

xp=numpy.linspace(numpy.min(data[:,0]),numpy.max(data[:,0]),100)

for i in range(npolyorders):
    f = numpy.polyfit(data[:,0], data[:,1], i)
    p=numpy.poly1d(f)
    plt.plot(xp,p(xp))
plt.legend(['%d'%i for i in range(npolyorders)])

# compute in-sample and out-of-sample error using LOO
loo=LeaveOneOut()
pred=numpy.zeros((n,npolyorders))
mean_trainerr=numpy.zeros(npolyorders)
prederr=numpy.zeros(npolyorders)

for i in range(npolyorders):
    ctr=0
    trainerr=numpy.zeros(n)
    for train,test in loo.split(range(data.shape[0])):
        f = numpy.polyfit(data[train,0], data[train,1], i)
        p=numpy.poly1d(f)
        trainerr[ctr]=numpy.sqrt(numpy.mean((data[train,1]-p(data[train,0]))**2))
        pred[test,i]=p(data[test,0])
        ctr+=1
    mean_trainerr[i]=numpy.mean(trainerr)
    prederr[i]=numpy.sqrt(numpy.mean((data[:,1]-pred[:,i])**2))
    

plt.plot(range(npolyorders),mean_trainerr)
plt.plot(range(npolyorders),prederr,color='red')
plt.xlabel('Polynomial order')
plt.ylabel('root mean squared error')
plt.legend(['training error','test error'],loc=9)
plt.plot([numpy.argmin(prederr),numpy.argmin(prederr)],
         [numpy.min(mean_trainerr),numpy.max(prederr)],'k--')
plt.text(0.5,numpy.max(mean_trainerr),'underfitting')
plt.text(4.5,numpy.max(mean_trainerr),'overfitting')

print('True order:',trueorder)
print('Order estimated by cross validation:',numpy.argmin(prederr))

def make_larger_dataset(beta,n,sd=1):
    X=numpy.random.randn(n,len(beta)) # design matrix
    beta=numpy.array(beta)
    y=X.dot(beta)+numpy.random.randn(n)*sd
    return(y-numpy.mean(y),X)
    

def compare_lr_lasso(n=100,nvars=20,n_splits=8,sd=1):
    beta=numpy.zeros(nvars)
    beta[0]=1
    beta[1]=-1
    y,X=make_larger_dataset(beta,100,sd=1)
    
    kf=KFold(n_splits=n_splits,shuffle=True)
    pred_lr=numpy.zeros(X.shape[0])
    coefs_lr=numpy.zeros((n_splits,X.shape[1]))
    pred_lasso=numpy.zeros(X.shape[0])
    coefs_lasso=numpy.zeros((n_splits,X.shape[1]))
    lr=LinearRegression()
    lasso=LassoCV()
    ctr=0
    for train,test in kf.split(X):
        Xtrain=X[train,:]
        Ytrain=y[train]
        lr.fit(Xtrain,Ytrain)
        lasso.fit(Xtrain,Ytrain)
        pred_lr[test]=lr.predict(X[test,:])
        coefs_lr[ctr,:]=lr.coef_
        pred_lasso[test]=lasso.predict(X[test,:])
        coefs_lasso[ctr,:]=lasso.coef_
        ctr+=1
    prederr_lr=numpy.sum((pred_lr-y)**2)
    prederr_lasso=numpy.sum((pred_lasso-y)**2)
    return [prederr_lr,prederr_lasso],numpy.mean(coefs_lr,0),numpy.mean(coefs_lasso,0),beta


nsims=100
prederr=numpy.zeros((nsims,2))
lrcoef=numpy.zeros((nsims,20))
lassocoef=numpy.zeros((nsims,20))

for i in range(nsims):
    prederr[i,:],lrcoef[i,:],lassocoef[i,:],beta=compare_lr_lasso()
    
print('mean sum of squared error:')
print('linear regression:',numpy.mean(prederr,0)[0])
print('lasso:',numpy.mean(prederr,0)[1])

coefs_df=pandas.DataFrame({'True value':beta,'Regression (mean)':numpy.mean(lrcoef,0),'Lasso (mean)':numpy.mean(lassocoef,0),
                          'Regression(stdev)':numpy.std(lrcoef,0),'Lasso(stdev)':numpy.std(lassocoef,0)})
coefs_df

nsims=100
prederr=numpy.zeros((nsims,2))
lrcoef=numpy.zeros((nsims,1000))
lassocoef=numpy.zeros((nsims,1000))

for i in range(nsims):
    prederr[i,:],lrcoef[i,:],lassocoef[i,:],beta=compare_lr_lasso(nvars=1000)
    
print('mean sum of squared error:')
print('linear regression:',numpy.mean(prederr,0)[0])
print('lasso:',numpy.mean(prederr,0)[1])



