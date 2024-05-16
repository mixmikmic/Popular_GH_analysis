get_ipython().run_line_magic('pylab', 'inline')
import seaborn, time
seaborn.set_style('whitegrid')

from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from pomegranate import *

def create_dataset(n_samples, n_dim, n_classes):
    """Create a random dataset with n_samples in each class."""
    
    X = numpy.concatenate([numpy.random.randn(n_samples, n_dim) + i*3 for i in range(n_classes)])
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
    
def evaluate_kmeans():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )

            pom = Kmeans(2)
            skl = KMeans(2, max_iter=1, n_init=1, precompute_distances=True, init=X[:2].copy())

            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
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

evaluate_kmeans()

def evaluate_models():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 1, 2 )

            pom = GeneralMixtureModel( NormalDistribution, n_components=2 )
            skl = GMM( n_components=3, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
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

evaluate_models()

def evaluate_models():
    sizes = numpy.arange(2, 21, dtype='int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))

    for i in range(m):
        for j, size in enumerate(sizes):        
            X, y = create_dataset(10000 / size, 1, size )

            pom = GeneralMixtureModel( NormalDistribution, n_components=size )
            skl = GMM( n_components=size, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
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
    plot(fit, predict, skl_error, pom_error, sizes, "number of components")

evaluate_models()

def evaluate_models():
    sizes = numpy.around( numpy.exp( numpy.arange(8, 16) ) ).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset( size, 5, 2 )
            
            pom = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=2)
            skl = GMM( n_components=2, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
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

evaluate_models()

numpy.set_printoptions(suppress=True, linewidth=200)

def evaluate_models():
    sizes = numpy.arange(2, 21).astype('int')
    n, m = sizes.shape[0], 20
    
    skl_predict, pom_predict = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_fit, pom_fit = numpy.zeros((m, n)), numpy.zeros((m, n))
    skl_error, pom_error = numpy.zeros((m, n)), numpy.zeros((m, n))
    
    for i in range(m):
        for j, size in enumerate(sizes):
            X, y = create_dataset(50000, size, 2)
            
            pom = GeneralMixtureModel(MultivariateGaussianDistribution, n_components=2)
            skl = GMM( n_components=2, n_iter=1 )
            
            # bench fit times
            tic = time.time()
            skl.fit( X )
            skl_fit[i, j] = time.time() - tic

            tic = time.time()
            pom.fit( X, max_iterations=1 )
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

evaluate_models()

