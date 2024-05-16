import os

import numpy as np

import theano
import theano.tensor as T

import time

def show_config():
    print("OMP_NUM_THREADS                       = %s" % 
           os.environ.get('OMP_NUM_THREADS','#CAREFUL : OMP_NUM_THREADS Not-defined!'))

    print("theano.config.device                  = %s" % theano.config.device)
    print("theano.config.floatX                  = %s" % theano.config.floatX)
    print("theano.config.blas.ldflags            = '%s'" % theano.config.blas.ldflags)
    print("theano.config.openmp                  = %s" % theano.config.openmp)
    print("theano.config.openmp_elemwise_minsize = %d" % theano.config.openmp_elemwise_minsize)

    # IDEA for pretty-printing : http://stackoverflow.com/questions/32026727/format-output-of-code-cell-with-markdown

def show_timing(iters=8, order='C'):
    M, N, K = 2000, 2000, 2000
    
    a = theano.shared(np.ones((M, N), dtype=theano.config.floatX, order=order))
    b = theano.shared(np.ones((N, K), dtype=theano.config.floatX, order=order))
    c = theano.shared(np.ones((M, K), dtype=theano.config.floatX, order=order))
    
    f = theano.function([], updates=[(c, 0.4 * c + 0.8 * T.dot(a, b))])
    
    if any([x.op.__class__.__name__ == 'Gemm' for x in f.maker.fgraph.toposort()]):
        c_impl = [hasattr(thunk, 'cthunk')
                  for node, thunk in zip(f.fn.nodes, f.fn.thunks)
                  if node.op.__class__.__name__ == "Gemm"]
        assert len(c_impl) == 1
        
        if c_impl[0]:
            impl = 'CPU (with direct Theano binding to blas)'
        else:
            impl = 'CPU (no direct Theano binding to blas, using numpy/scipy)'
            
    elif any([x.op.__class__.__name__ == 'GpuGemm' for x in
              f.maker.fgraph.toposort()]):
        impl = 'GPU'
        
    else:
        impl = 'ERROR, unable to tell if Theano used the cpu or the gpu:\n'
        impl += str(f.maker.fgraph.toposort())
    
    print("\nRunning operations using              : %s" % impl)
    
    t0 = time.time()
    for i in range(iters):
        f()
    if False:
        theano.sandbox.cuda.synchronize()
        
    print("Time taken for each of %2d iterations  : %.0f msec" % (iters, 1000.*(time.time()-t0)/iters))

show_config()
show_timing()

#os.environ['OMP_NUM_THREADS']="1"
#os.environ['OMP_NUM_THREADS']="4"
#theano.config.floatX = 'float64'
theano.config.floatX = 'float32'
theano.config.openmp = False
#theano.config.openmp = True
#theano.config.blas.ldflags = ''
#theano.config.blas.ldflags = '-L/lib64/atlas -lsatlas'
theano.config.blas.ldflags = '-L/lib64/atlas -ltatlas'

show_config()
show_timing()



