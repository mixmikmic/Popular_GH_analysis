get_ipython().run_line_magic('pylab', 'inline')
import hmmlearn, pomegranate, time, seaborn
from hmmlearn.hmm import *
from pomegranate import *
seaborn.set_style('whitegrid')

print "hmmlearn version {}".format(hmmlearn.__version__)
print "pomegranate version {}".format(pomegranate.__version__)

def initialize_components(n_components, n_dims, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.abs(numpy.random.randn(n_components, n_components))
    transmat = (transmat.T / transmat.sum( axis=1 )).T

    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    means = numpy.random.randn(n_components, n_dims)
    covars = numpy.ones((n_components, n_dims))
    
    seqs = numpy.zeros((n_seqs, n_components, n_dims))
    for i in range(n_seqs):
        seqs[i] = means + numpy.random.randn(n_components, n_dims)
        
    return transmat, start_probs, means, covars, seqs

def hmmlearn_model(transmat, start_probs, means, covars):
    """Return a hmmlearn model."""

    model = GaussianHMM(n_components=transmat.shape[0], covariance_type='diag', n_iter=1, tol=1e-8)
    model.startprob_ = start_probs
    model.transmat_ = transmat
    model.means_ = means
    model._covars_ = covars
    return model

def pomegranate_model(transmat, start_probs, means, covars):
    """Return a pomegranate model."""
    
    states = [ MultivariateGaussianDistribution( means[i], numpy.eye(means.shape[1]) ) for i in range(transmat.shape[0]) ]
    model = HiddenMarkovModel.from_matrix(transmat, states, start_probs, merge='None')
    return model

def evaluate_models(n_dims, n_seqs):
    hllp, plp = [], []
    hlv, pv = [], []
    hlm, pm = [], []
    hls, ps = [], []
    hlt, pt = [], []

    for i in range(10, 112, 10):
        transmat, start_probs, means, covars, seqs = initialize_components(i, n_dims, n_seqs)
        model = hmmlearn_model(transmat, start_probs, means, covars)

        tic = time.time()
        for seq in seqs:
            model.score(seq)
        hllp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        hlv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        hlm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs.reshape(n_seqs*i, n_dims), lengths=[i]*n_seqs)
        hlt.append( time.time() - tic )

        model = pomegranate_model(transmat, start_probs, means, covars)

        tic = time.time()
        for seq in seqs:
            model.log_probability(seq)
        plp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        pv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        pm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs, max_iterations=1, verbose=False)
        pt.append( time.time() - tic )

    plt.figure( figsize=(12, 8))
    plt.xlabel("# Components", fontsize=12 )
    plt.ylabel("pomegranate is x times faster", fontsize=12 )
    plt.plot( numpy.array(hllp) / numpy.array(plp), label="Log Probability")
    plt.plot( numpy.array(hlv) / numpy.array(pv), label="Viterbi")
    plt.plot( numpy.array(hlm) / numpy.array(pm), label="Maximum A Posteriori")
    plt.plot( numpy.array(hlt) / numpy.array(pt), label="Training")
    plt.xticks( xrange(11), xrange(10, 112, 10), fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.legend( fontsize=12 )

evaluate_models(10, 50)

def initialize_components(n_components, n_dims, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.zeros((n_components, n_components))
    transmat[-1, -1] = 1
    for i in range(n_components-1):
        transmat[i, i] = 1
        transmat[i, i+1] = 1
    transmat[ transmat < 0 ] = 0
    transmat = (transmat.T / transmat.sum( axis=1 )).T

    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    means = numpy.random.randn(n_components, n_dims)
    covars = numpy.ones((n_components, n_dims))
    
    seqs = numpy.zeros((n_seqs, n_components, n_dims))
    for i in range(n_seqs):
        seqs[i] = means + numpy.random.randn(n_components, n_dims)
        
    return transmat, start_probs, means, covars, seqs

evaluate_models(10, 50)

def initialize_components(n_components, n_seqs):
    """
    Initialize a transition matrix for a model with a fixed number of components,
    for Gaussian emissions with a certain number of dimensions, and a data set
    with a certain number of sequences.
    """
    
    transmat = numpy.zeros((n_components, n_components))
    transmat[-1, -1] = 1
    for i in range(n_components-1):
        transmat[i, i] = 1
        transmat[i, i+1] = 1
    transmat[ transmat < 0 ] = 0
    transmat = (transmat.T / transmat.sum( axis=1 )).T
    
    start_probs = numpy.abs( numpy.random.randn(n_components) )
    start_probs /= start_probs.sum()

    dists = numpy.abs(numpy.random.randn(n_components, 4))
    dists = (dists.T / dists.T.sum(axis=0)).T
    
    seqs = numpy.random.randint(0, 4, (n_seqs, n_components*2, 1))
    return transmat, start_probs, dists, seqs

def hmmlearn_model(transmat, start_probs, dists):
    """Return a hmmlearn model."""

    model = MultinomialHMM(n_components=transmat.shape[0], n_iter=1, tol=1e-8)
    model.startprob_ = start_probs
    model.transmat_ = transmat
    model.emissionprob_ = dists
    return model

def pomegranate_model(transmat, start_probs, dists):
    """Return a pomegranate model."""
    
    states = [ DiscreteDistribution({ 'A': d[0],
                                      'C': d[1],
                                      'G': d[2], 
                                      'T': d[3] }) for d in dists ]
    model = HiddenMarkovModel.from_matrix(transmat, states, start_probs, merge='None')
    return model

def evaluate_models(n_seqs):
    hllp, plp = [], []
    hlv, pv = [], []
    hlm, pm = [], []
    hls, ps = [], []
    hlt, pt = [], []

    dna = 'ACGT'
    
    for i in range(10, 112, 10):
        transmat, start_probs, dists, seqs = initialize_components(i, n_seqs)
        model = hmmlearn_model(transmat, start_probs, dists)

        tic = time.time()
        for seq in seqs:
            model.score(seq)
        hllp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        hlv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        hlm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs.reshape(n_seqs*i*2, 1), lengths=[i*2]*n_seqs)
        hlt.append( time.time() - tic )

        model = pomegranate_model(transmat, start_probs, dists)
        seqs = [[dna[i] for i in seq] for seq in seqs]

        tic = time.time()
        for seq in seqs:
            model.log_probability(seq)
        plp.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict(seq)
        pv.append( time.time() - tic )

        tic = time.time()
        for seq in seqs:
            model.predict_proba(seq)
        pm.append( time.time() - tic )    
        
        tic = time.time()
        model.fit(seqs, max_iterations=1, verbose=False)
        pt.append( time.time() - tic )

    plt.figure( figsize=(12, 8))
    plt.xlabel("# Components", fontsize=12 )
    plt.ylabel("pomegranate is x times faster", fontsize=12 )
    plt.plot( numpy.array(hllp) / numpy.array(plp), label="Log Probability")
    plt.plot( numpy.array(hlv) / numpy.array(pv), label="Viterbi")
    plt.plot( numpy.array(hlm) / numpy.array(pm), label="Maximum A Posteriori")
    plt.plot( numpy.array(hlt) / numpy.array(pt), label="Training")
    plt.xticks( xrange(11), xrange(10, 112, 10), fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.legend( fontsize=12 )

evaluate_models(50)

