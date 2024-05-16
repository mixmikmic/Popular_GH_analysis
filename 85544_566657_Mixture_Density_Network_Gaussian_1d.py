from matplotlib import pyplot
get_ipython().magic('matplotlib inline')
import torch
import torch.utils.data
import numpy
from torch.autograd import Variable
import IPython
import itertools
import seaborn

n_train = 1000
batch_size = 32
class DS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
        self.y = torch.rand(n)*21-10.5
        self.x = torch.sin(0.75*self.y)*7.0+self.y*0.5+torch.randn(n)
    def __len__(self):
        return self.n
    def __getitem__(self,i):
        return (self.x[i],self.y[i])

train_ds = DS(n_train)
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
pyplot.show()
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

class GaussianMixture1d(torch.nn.Module):
    def __init__(self, n_in, n_mixtures, eps=0):
        super(GaussianMixture1d, self).__init__()
        self.n_in = n_in
        self.eps = eps
        self.n_mixtures = n_mixtures        
        self.lin = torch.nn.Linear(n_in, 3*n_mixtures)
        self.log_2pi = numpy.log(2*numpy.pi)

    def params(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x input
        p = self.lin(inp)
        pi = torch.nn.functional.softmax(p[:,:self.n_mixtures]*(1+pi_bias)) # mixture weights (probability weights)
        mu = p[:,self.n_mixtures:2*self.n_mixtures] # means of the 1d gaussians
        sigma = (p[:,2*self.n_mixtures:]-std_bias).exp() # stdevs of the 1d gaussians
        sigma = sigma+self.eps
        return pi,mu,sigma

    def forward(self, inp, x):
        # x = batch x 3 (=movement x,movement y,end of stroke)
        # loss, negative log likelihood
        pi,mu,sigma = self.params(inp)
        log_normal_likelihoods =  -0.5*((x.unsqueeze(1)-mu) / sigma)**2-0.5*self.log_2pi-torch.log(sigma) # batch x n_mixtures
        log_weighted_normal_likelihoods = log_normal_likelihoods+pi.log() # batch x n_mixtures
        maxes,_ = log_weighted_normal_likelihoods.max(1)
        mixture_log_likelihood = (log_weighted_normal_likelihoods-maxes.unsqueeze(1)).exp().sum(1).log()+maxes # log-sum-exp with stabilisation
        neg_log_lik = -mixture_log_likelihood
        return neg_log_lik

    def predict(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x n_in
        pi,mu,sigma = self.params(inp, pi_bias=pi_bias, std_bias=std_bias)
        x = inp.data.new(inp.size(0)).normal_()
        mixture = pi.data.multinomial(1)       # batch x 1 , index to the mixture component
        sel_mu = mu.data.gather(1, mixture).squeeze(1)
        sel_sigma = sigma.data.gather(1, mixture).squeeze(1)
        x = x*sel_sigma+sel_mu
        return Variable(x)

class Model(torch.nn.Module):
    def __init__(self, n_inp = 1, n_hid = 24, n_mixtures = 24):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(n_inp, n_hid)
        self.mix = GaussianMixture1d(n_hid, n_mixtures)
    def forward(self, inp, x):
        h = torch.tanh(self.lin(inp))
        l = self.mix(h, x)
        return l.mean()
    def predict(self, inp, pi_bias=0, std_bias=0):
        h = torch.tanh(self.lin(inp))
        return self.mix.predict(h, std_bias=std_bias, pi_bias=pi_bias)

m = Model(1, 32, 20)     
opt = torch.optim.Adam(m.parameters(), 0.001)
m.cuda()
losses = []
for epoch in range(2000):
    thisloss  = 0
    for i,(x,y) in enumerate(train_dl):
        x = Variable(x.float().unsqueeze(1).cuda())
        y = Variable(y.float().cuda())
        opt.zero_grad()
        loss = m(x, y)
        loss.backward()
        thisloss += loss.data[0]/len(train_dl)
        opt.step()
    losses.append(thisloss)
    if epoch % 10 == 0:
        IPython.display.clear_output(wait=True)
        print (epoch, loss.data[0])
        x = Variable(torch.rand(1000,1).cuda()*30-15)
        y = m.predict(x)
        y2 = m.predict(x, std_bias=10)
        pyplot.subplot(1,2,1)
        pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
        pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y.data.cpu().numpy(),facecolor='r', s=3)
        pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y2.data.cpu().numpy(),facecolor='g', s=3)
        pyplot.subplot(1,2,2)
        pyplot.title("loss")
        pyplot.plot(losses)
        pyplot.show()

class G(torch.nn.Module):
    def __init__(self, n_random=2, n_hidden=50):
        super(G, self).__init__()
        self.n_random = n_random
        self.l1 = torch.nn.Linear(n_random,n_hidden)
        self.l2 = torch.nn.Linear(n_hidden,n_hidden)
        #self.l2b = torch.nn.Linear(n_hidden,n_hidden)
        self.l3 = torch.nn.Linear(n_hidden,2)
    def forward(self, batch_size=32):
        x = Variable(self.l1.weight.data.new(batch_size, self.n_random).normal_())
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        #x = torch.nn.functional.relu(self.l2b(x))
        x = self.l3(x)
        return x

class D(torch.nn.Module):
    def __init__(self, lam=10.0, n_hidden=50):
        super(D, self).__init__()
        self.l1 = torch.nn.Linear(2,n_hidden)
        self.l2 = torch.nn.Linear(n_hidden,n_hidden)
        self.l3 = torch.nn.Linear(n_hidden,1)
        self.one = torch.FloatTensor([1]).cuda()
        self.mone = torch.FloatTensor([-1]).cuda()
        self.lam = lam
    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x
    def slogan_loss_and_backward(self, real, fake):
        self.zero_grad()
        f_real = self(real.detach())
        f_real_sum = f_real.sum()
        f_real_sum.backward(self.one, retain_graph=True)
        f_fake = self(fake.detach())
        f_fake_sum = f_fake.sum()
        f_fake_sum.backward(self.mone, retain_graph=True)
        f_mean = (f_fake_sum+f_real_sum)
        f_mean.abs().backward(retain_graph=True)
        dist = ((real.view(real.size(0),-1).unsqueeze(0)-fake.view(fake.size(0),-1).unsqueeze(1))**2).sum(2)**0.5
        f_diff = (f_real.unsqueeze(0)-f_fake.unsqueeze(1)).squeeze(2).abs()
        lip_dists = f_diff/(dist+1e-6)
        lip_penalty = (self.lam * (lip_dists.clamp(min=1)-1)**2).sum()
        lip_penalty.backward()
        return f_real_sum.data[0],f_fake_sum.data[0],lip_penalty.data[0], lip_dists.data.mean()

d = D()
d.cuda()
g = G(n_random=256, n_hidden=128)
g.cuda()
opt_d = torch.optim.Adam(d.parameters(), lr=1e-3)
opt_g = torch.optim.Adam(g.parameters(), lr=1e-3)

for p in itertools.chain(d.parameters(), g.parameters()):
    if p.data.dim()>1:
        torch.nn.init.orthogonal(p, 2.0)

def endless(dl):
    while True:
        for i in dl:
            yield i

batch_size=256
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
train_iter = endless(train_dl)

for i in range(30001):
    if i>0 and i%10000 == 0:
        for pg in opt_g.param_groups:
            pg['lr'] /= 3
        for pg in opt_d.param_groups:
            pg['lr'] /= 3
        print ("new lr",pg['lr'])
    for j in range(100 if i < 10 or i % 100 == 0 else 10):
        g.eval()
        d.train()
        for p in d.parameters():
            p.requires_grad = True
        real_x,real_y = next(train_iter)
        real = Variable(torch.stack([real_x.float().cuda(), real_y.float().cuda()],dim=1))
        fake = g(batch_size=batch_size)
        l_r, l_f, l_lip, lip_mean = d.slogan_loss_and_backward(real, fake)
        opt_d.step()
    if i % 100 == 0:
        print ("f_r:",l_r,"f_fake:",l_f, "lip loss:", l_lip, "lip_mean", lip_mean)
    g.train()
    d.eval()
    for p in d.parameters():
        p.requires_grad = False
    g.zero_grad()
    fake = g(batch_size=batch_size)
    f = d(fake)
    fsum = f.sum()
    fsum.backward()
    opt_g.step()
    if i % 1000 == 0:
        IPython.display.clear_output(wait=True)
        print (i)
        fake = g(batch_size=10000)
        fd = fake.data.cpu().numpy()
        pyplot.figure(figsize=(15,5))
        pyplot.subplot(1,3,1)
        pyplot.title("Generated Density")
        seaborn.kdeplot(fd[:,0], fd[:,1], shade=True, cmap='Greens', bw=0.01)
        pyplot.subplot(1,3,2)
        pyplot.title("Data Density Density")
        seaborn.kdeplot(train_ds.x.numpy(),train_ds.y.numpy(), shade=True, cmap='Greens', bw=0.01)
        pyplot.subplot(1,3,3)
        pyplot.title("Data (blue) and generated (red)")
        pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
        pyplot.scatter(fd[:,0], fd[:,1], facecolor='r',s=3, alpha=0.5)
        pyplot.show()

N_dim = 100
x_test = torch.linspace(-20,20, N_dim)
y_test = torch.linspace(-15,15, N_dim)
x_test = (x_test.unsqueeze(0)*torch.ones(N_dim,1)).view(-1)
y_test = (y_test.unsqueeze(1)*torch.ones(1,N_dim)).view(-1)
xy_test = Variable(torch.stack([x_test, y_test], dim=1).cuda())
f_test = d(xy_test)
pyplot.imshow(f_test.data.view(N_dim,N_dim).cpu().numpy(), origin='lower', cmap=pyplot.cm.gist_heat_r, extent=(-20,20,-15,15))
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), facecolor='b', s=2);



