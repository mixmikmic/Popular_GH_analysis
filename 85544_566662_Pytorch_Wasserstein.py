import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function
from torchvision import datasets, transforms

class WassersteinLossVanilla(Function):
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossVanilla,self).__init__()
        
        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None
        
    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        nbatch = pred.size(0)
        
        u = self.cost.new(nbatch, self.na).fill_(1.0/self.na)
        
        for i in range(self.sinkhorn_iter):
            v = target/(torch.mm(u,self.K.t())) # double check K vs. K.t() here and next line
            u = pred/(torch.mm(v,self.K))
            #print ("stability at it",i, "u",(u!=u).sum(),u.max(),"v", (v!=v).sum(), v.max())
            if (u!=u).sum()>0 or (v!=v).sum()>0 or u.max()>1e9 or v.max()>1e9: # u!=u is a test for NaN...
                # we have reached the machine precision
                # come back to previous solution and quit loop
                raise Exception(str(('Warning: numerical errrors',i+1,"u",(u!=u).sum(),u.max(),"v",(v!=v).sum(),v.max())))

        loss = (u*torch.mm(v,self.KM.t())).mean(0).sum() # double check KM vs KM.t()...
        grad = self.lam*u.log()/nbatch # check whether u needs to be transformed        
        grad = grad-torch.mean(grad,dim=1).expand_as(grad)
        grad = grad-torch.mean(grad,dim=1).expand_as(grad) # does this help over only once?
        self.stored_grad = grad

        dist = self.cost.new((loss,))
        return dist
    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        return self.stored_grad*grad_output[0],None

class WassersteinLossStab(Function):
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossStab,self).__init__()
        
        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None
        
    def forward(self, pred, target):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        batch_size = pred.size(0)
        
        log_a, log_b = torch.log(pred), torch.log(target)
        log_u = self.cost.new(batch_size, self.na).fill_(-numpy.log(self.na))
        log_v = self.cost.new(batch_size, self.nb).fill_(-numpy.log(self.nb))
        
        for i in range(self.sinkhorn_iter):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u-log_u_max.expand_as(log_u))
            log_v = log_b - torch.log(torch.mm(self.K.t(),u_stab.t()).t()) - log_u_max.expand_as(log_v)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
            log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max.expand_as(log_u)

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v-log_v_max.expand_as(log_v))
        logcostpart1 = torch.log(torch.mm(self.KM,v_stab.t()).t())+log_v_max.expand_as(log_u)
        wnorm = torch.exp(log_u+logcostpart1).mean(0).sum() # sum(1) for per item pair loss...
        grad = log_u*self.lam
        grad = grad-torch.mean(grad,dim=1).expand_as(grad)
        grad = grad-torch.mean(grad,dim=1).expand_as(grad) # does this help over only once?
        grad = grad/batch_size
        
        self.stored_grad = grad

        return self.cost.new((wnorm,))
    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        #print (self.stored_grad, grad_output)
        res = grad_output.new()
        res.resize_as_(self.stored_grad).copy_(self.stored_grad)
        if grad_output[0] != 1:
            res.mul_(grad_output[0])
        return res,None

import ot
import numpy
from matplotlib import pyplot
get_ipython().magic('matplotlib inline')

# test problem from Python Optimal Transport
n=100
a=ot.datasets.get_1D_gauss(n,m=20,s=10).astype(numpy.float32)
b=ot.datasets.get_1D_gauss(n,m=60,s=30).astype(numpy.float32)
c=ot.datasets.get_1D_gauss(n,m=40,s=20).astype(numpy.float32)
a64=ot.datasets.get_1D_gauss(n,m=20,s=10).astype(numpy.float64)
b64=ot.datasets.get_1D_gauss(n,m=60,s=30).astype(numpy.float64)
c64=ot.datasets.get_1D_gauss(n,m=40,s=20).astype(numpy.float64)
# distance function
x=numpy.arange(n,dtype=numpy.float32)
M=(x[:,numpy.newaxis]-x[numpy.newaxis,:])**2
M/=M.max()
x64=numpy.arange(n,dtype=numpy.float64)
M64=(x64[:,numpy.newaxis]-x64[numpy.newaxis,:])**2
M64/=M64.max()

transp = ot.bregman.sinkhorn(a,b,M,reg=1e-3)
transp2 = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-3)

(transp*M).sum(), (transp2*M).sum()

cabt = Variable(torch.from_numpy(numpy.stack((c,a,b),axis=0)))
abct = Variable(torch.from_numpy(numpy.stack((a,b,c),axis=0)))

lossvanilla = WassersteinLossVanilla(torch.from_numpy(M), lam=0.1)
loss = lossvanilla
losses = loss(cabt,abct), loss(cabt[:1],abct[:1]), loss(cabt[1:2],abct[1:2]), loss(cabt[2:],abct[2:])
sum(losses[1:])/3, losses

loss = WassersteinLossStab(torch.from_numpy(M), lam=0.1)
losses = loss(cabt,abct), loss(cabt[:1],abct[:1]), loss(cabt[1:2],abct[1:2]), loss(cabt[2:],abct[2:])
sum(losses[1:])/3, losses

transp3 = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-2)
loss = WassersteinLossStab(torch.from_numpy(M), lam=0.01)
(transp3*M).sum(), loss(cabt[1:2],abct[1:2]).data[0]

theloss = WassersteinLossStab(torch.from_numpy(M), lam=0.01, sinkhorn_iter=50)
cabt = Variable(torch.from_numpy(numpy.stack((c,a,b),axis=0)))
abct = Variable(torch.from_numpy(numpy.stack((a,b,c),axis=0)),requires_grad=True)
lossv1 = theloss(abct,cabt)
lossv1.backward()
grv = abct.grad
epsilon = 1e-5
abctv2 = Variable(abct.data-epsilon*grv.data, requires_grad=True)
lossv2 = theloss(abctv2, cabt)
lossv2.backward()
grv2 = abctv2.grad
(lossv1.data-lossv2.data)/(epsilon*((0.5*(grv.data+grv2.data))**2).sum()) # should be around 1

def sinkhorn(a,b, M, reg, numItermax = 1000, stopThr=1e-9, verbose=False, log=False):
    # seems to explode terribly fast with 32 bit floats...
    if a is None:
        a = M.new(M.size(0)).fill_(1/m.size(0))
    if b is None:
        b = M.new(M.size(0)).fill_(1/M.size(1))

    # init data
    Nini = a.size(0)
    Nfin = b.size(0)

    cpt = 0
    if log:
        log={'err':[]}

    # we assume that no distances are null except those of the diagonal of distances
    u = M.new(Nfin).fill_(1/Nfin)
    v = M.new(Nfin).fill_(1/Nfin)
    uprev=M.new(Nini).zero_()
    vprev=M.new(Nini).zero_()

    K = torch.exp(-M/reg)

    Kp = K/(a[:,None].expand_as(K))
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        Kt_dot_u = torch.mv(K.t(),u)
        if (Kt_dot_u==0).sum()>0 or (u!=u).sum()>0 or (v!=v).sum()>0: # u!=u is a test for NaN...
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break
        uprev = u
        vprev = v
        v = b/Kt_dot_u
        u = 1./torch.mv(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp =   (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K))
            err = torch.dist(transp.sum(0),b)**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))
        cpt = cpt +1
    if log:
        log['u']=u
        log['v']=v
    #print 'err=',err,' cpt=',cpt
    if log:
        return (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K)),log
    else:
        return (u[:,None].expand_as(K))*K*(v[None,:].expand_as(K))

# test 32 bit vs. 64 bit for unstabilized
typ = numpy.float64
dist_torch64 =  sinkhorn(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
typ = numpy.float32
dist_torch32 =  sinkhorn(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
dist_pot     = ot.bregman.sinkhorn(a,b,M,reg=1e-3)
numpy.abs(dist_torch64.numpy()-dist_pot).max(), numpy.abs(dist_torch32.numpy()-dist_pot).max()

def sinkhorn_stabilized(a,b, M, reg, numItermax = 1000,tau=1e3, stopThr=1e-9,
                        warmstart=None, verbose=False,print_period=20, log=False):
    if a is None:
        a = M.new(m.size(0)).fill_(1/m.size(0))
    if b is None:
        b = M.new(m.size(0)).fill_(1/m.size(1))

    # init data
    na = a.size(0)
    nb = b.size(0)

    cpt = 0
    if log:
        log={'err':[]}


    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha,beta=M.new(na).zero_(),M.new(nb).zero_()
    else:
        alpha,beta=warmstart
    u,v = M.new(na).fill_(1/na),M.new(nb).fill_(1/nb)
    uprev,vprev=M.new(na).zero_(),M.new(nb).zero_()

    def get_K(alpha,beta):
        """log space computation"""
        return torch.exp(-(M-alpha[:,None].expand_as(M)-beta[None,:].expand_as(M))/reg)

    def get_Gamma(alpha,beta,u,v):
        """log space gamma computation"""
        return torch.exp(-(M-alpha[:,None].expand_as(M)-beta[None,:].expand_as(M))/reg+torch.log(u)[:,None].expand_as(M)+torch.log(v)[None,:].expand_as(M))

    K=get_K(alpha,beta)
    transp = K
    loop=True
    cpt = 0
    err=1
    while loop:

        if  u.abs().max()>tau or  v.abs().max()>tau:
            alpha, beta = alpha+reg*torch.log(u), beta+reg*torch.log(v)
            u,v = M.new(na).fill_(1/na),M.new(nb).fill_(1/nb)
            K=get_K(alpha,beta)

        uprev = u
        vprev = v
        
        Kt_dot_u = torch.mv(K.t(),u)
        v = b/Kt_dot_u
        u = a/torch.mv(K,v)

        if cpt%print_period==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = get_Gamma(alpha,beta,u,v)
            err = torch.dist(transp.sum(0),b)**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt%(print_period*20) ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))


        if err<=stopThr:
            loop=False

        if cpt>=numItermax:
            loop=False


        if (Kt_dot_u==0).sum()>0 or (u!=u).sum()>0 or (v!=v).sum()>0: # u!=u is a test for NaN...
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev
            break

        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt
    if log:
        log['logu']=alpha/reg+torch.log(u)
        log['logv']=beta/reg+torch.log(v)
        log['alpha']=alpha+reg*torch.log(u)
        log['beta']=beta+reg*torch.log(v)
        log['warmstart']=(log['alpha'],log['beta'])
        return get_Gamma(alpha,beta,u,v),log
    else:
        return get_Gamma(alpha,beta,u,v)

# test 32 bit vs. 64 bit for stabilized
typ = numpy.float64
dist_torch64 =  sinkhorn_stabilized(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
typ = numpy.float32
dist_torch32 =  sinkhorn_stabilized(torch.from_numpy(a.astype(typ)),torch.from_numpy(b.astype(typ)),
                   torch.from_numpy(M.astype(typ)),reg=1e-3)
dist_pot     = ot.bregman.sinkhorn_stabilized(a,b,M,reg=1e-3)
numpy.abs(dist_torch64.numpy()-dist_pot).max(), numpy.abs(dist_torch32.numpy()-dist_pot).max()



