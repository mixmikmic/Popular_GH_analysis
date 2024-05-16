get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')

from sklearn.naive_bayes import GaussianNB
from pomegranate import *

def create_dataset(n_samples, n_dim, n_classes):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.randn(n_samples, n_dim) + i for i in range(n_classes)])
    y = numpy.concatenate([numpy.zeros(n_samples) + i for i in range(n_classes)])
    return X, y

def plot( fit, predict, skl_error, pom_error, sizes, xlabel ):
    """Plot the results."""
    
    idx = numpy.arange(fit.shape[1])
    
    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='c', label="Fitting")
    plt.plot( predict.mean(axis=0), c='m', label="Prediction")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )
    
    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, predict.min(axis=0), predict.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('pomegranate is x times faster', fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()
    
    
    plt.figure( figsize=(14, 4))
    plt.plot( 1 - skl_error.mean(axis=0), alpha=0.5, c='c', label="sklearn accuracy" )
    plt.plot( 1 - pom_error.mean(axis=0), alpha=0.5, c='m', label="pomegranate accuracy" )
    
    plt.fill_between( idx, 1-skl_error.min(axis=0), 1-skl_error.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, 1-pom_error.min(axis=0), 1-pom_error.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14) 
    plt.show()
    
def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")

skl = GaussianNB()
pom = NaiveBayes( NormalDistribution )
evaluate_models( skl, pom )

def evaluate_models( skl, pom ):
    sizes = numpy.arange(2, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( 50000 / size, 1, size )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "number of classes")

skl = GaussianNB()
pom = NaiveBayes( NormalDistribution )
evaluate_models( skl, pom )

X, y = create_dataset( 50000, 1, 2 )
skl = GaussianNB()
skl.fit(X, y)

pom = NaiveBayes( NormalDistribution )
pom.fit(X, y)

get_ipython().run_line_magic('timeit', 'skl.predict(X)')
get_ipython().run_line_magic('timeit', 'pom.predict(X)')

def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 5, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "samples per component")

skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )

def evaluate_models( skl, pom ):
    sizes = numpy.arange(3, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( 50000, size, 2 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X, y )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, y )
            pom_fit[i, j] = time.time() - tic

            # bench predict times
            tic = time.time()
            skl_predictions = skl.predict( X )
            skl_predict[i, j] = time.time() - tic

            tic = time.time()
            pom_predictions = pom.predict( X )
            pom_predict[i, j] = time.time() - tic
        
            # check number wrong
            skl_e = (y != skl_predictions).mean()
            pom_e = (y != pom_predictions).mean()

            skl_error[i, j] = min(skl_e, 1-skl_e)
            pom_error[i, j] = min(pom_e, 1-pom_e)
    
    fit = skl_fit / pom_fit
    predict = skl_predict / pom_predict
    
    plot(fit, predict, skl_error, pom_error, sizes, "dimensions")

skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )

def evaluate_models( skl, pom ):
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_time, pom_time = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            for l in range(5):
                X, y = create_dataset( size, 5, 2 )

                tic = time.time()
                skl.partial_fit( X, y, classes=[0, 1] )
                skl_time[i, j] += time.time() - tic

                tic = time.time()
                pom.summarize( X, y )
                pom_time[i, j] += time.time() - tic

            tic = time.time()
            pom.from_summaries()
            pom_time[i, j] += time.time() - tic

            skl_predictions = skl.predict( X )
            pom_predictions = pom.predict( X )

            skl_error[i, j] = ( y != skl_predictions ).mean()
            pom_error[i, j] = ( y != pom_predictions ).mean()
    
    fit = skl_time / pom_time
    idx = numpy.arange(fit.shape[1])
    
    plt.figure( figsize=(14, 4))
    plt.plot( fit.mean(axis=0), c='c', label="Fitting")
    plt.plot( [0, fit.shape[1]], [1, 1], c='k', label="Baseline" )
    plt.fill_between( idx, fit.min(axis=0), fit.max(axis=0), color='c', alpha=0.3 )
    
    plt.xticks(idx, sizes, rotation=65, fontsize=14)
    plt.xlabel('{}'.format(xlabel), fontsize=14)
    plt.ylabel('pomegranate is x times faster', fontsize=14)
    plt.legend(fontsize=12, loc=4)
    plt.show()
    
    
    plt.figure( figsize=(14, 4))
    plt.plot( 1 - skl_error.mean(axis=0), alpha=0.5, c='c', label="sklearn accuracy" )
    plt.plot( 1 - pom_error.mean(axis=0), alpha=0.5, c='m', label="pomegranate accuracy" )
    
    plt.fill_between( idx, 1-skl_error.min(axis=0), 1-skl_error.max(axis=0), color='c', alpha=0.3 )
    plt.fill_between( idx, 1-pom_error.min(axis=0), 1-pom_error.max(axis=0), color='m', alpha=0.3 )
    
    plt.xticks( idx, sizes, rotation=65, fontsize=14)
    plt.xlabel( '{}'.format(xlabel), fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14) 
    plt.show()

skl = GaussianNB()
pom = NaiveBayes( MultivariateGaussianDistribution )
evaluate_models( skl, pom )

