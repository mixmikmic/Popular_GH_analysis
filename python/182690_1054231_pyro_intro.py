get_ipython().system('pip install pyro-ppl')

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist

mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)

dist.normal.log_pdf(x, mu, sigma)

mu = pyro.param("mu", Variable(torch.zeros(1), requires_grad=True))
print(mu)

import torch.nn as nn

z_dim=20
hidden_dim=100

nn_decoder = nn.Sequential(
    nn.Linear(z_dim, hidden_dim), 
    nn.Softplus(), 
    nn.Linear(hidden_dim, 784), 
    nn.Sigmoid()
)

# import helper functions for Variables with requires_grad=False
from pyro.util import ng_zeros, ng_ones 

def model(x):
    batch_size=x.size(0)
    # register the decoder with Pyro (in particular all its parameters)
    pyro.module("decoder", nn_decoder)  
    # sample the latent code z
    z = pyro.sample("z", dist.normal,   
                    ng_zeros(batch_size, z_dim), 
                    ng_ones(batch_size, z_dim))
    # decode z into bernoulli probabilities
    bern_prob = nn_decoder(z)          
    # observe the mini-batch of sampled images
    return pyro.sample("x", dist.bernoulli, bern_prob, obs=x) 

class Encoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=100):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # define the forward computation on the image x
        # first compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma

nn_encoder = Encoder()

def guide(x):
    # register the encoder with Pyro
    pyro.module("encoder", nn_encoder)
    # encode the mini-batch of images x
    mu_z, sig_z = nn_encoder(x)
    # sample and return the latent code z
    return pyro.sample("z", dist.normal, mu_z, sig_z)

from pyro.optim import Adam
optimizer = Adam({"lr": 1.0e-3})

from pyro.infer import SVI
svi = SVI(model, guide, optimizer, loss="ELBO")

import torchvision.datasets as dset
import torchvision.transforms as transforms

batch_size=250
trans = transforms.ToTensor()
train_set = dset.MNIST(root='./mnist_data', train=True, 
                       transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batch_size,
                                           shuffle=True)

num_epochs = 3

for epoch in range(num_epochs):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # wrap the mini-batch of images in a PyTorch Variable
        x = Variable(x.view(-1, 784))
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # report training diagnostics
    normalizer = len(train_loader.dataset)
    print("[epoch %03d]  average training ELBO: %.4f" % (epoch, -epoch_loss / normalizer))

def geom(num_trials=0, bern_prob=0.5):
    p = Variable(torch.Tensor([bern_prob]))
    x = pyro.sample('x{}'.format(num_trials), dist.bernoulli, p)
    if x.data[0] == 1:
        return num_trials  # terminate recursion
    else:
        return geom(num_trials + 1, bern_prob)  # continue recursion

# let's draw 15 samples 
for _ in range(15):
    print("%d  " % geom()),

for _ in range(15):
    print("%d  " % geom(bern_prob=0.1)),

from torch.nn.functional import relu, sigmoid, grid_sample, affine_grid

z_dim=50

# this decodes latents z into (bernoulli pixel intensities for)
# 20x20 sized objects
class Decoder(nn.Module):
    def __init__(self, hidden_dim=200):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(z_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 20*20)

    def forward(self, z_what):
        h = relu(self.l1(z_what))
        return sigmoid(self.l2(h))

decoder = Decoder()

# define the prior probabilities for our random variables
z_where_prior_mu = Variable(torch.Tensor([3, 0, 0]))
z_where_prior_sigma = Variable(torch.Tensor([0.1, 1, 1]))
z_what_prior_mu = ng_zeros(50)
z_what_prior_sigma = ng_ones(50)

def expand_z_where(z_where):
    # Takes 3-dimensional vectors, and massages them into 
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    expansion_indices = Variable(torch.LongTensor([1, 0, 2, 0, 1, 3]))
    out = torch.cat((ng_zeros([1, 1]).expand(n, 1), z_where), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

# takes the object generated by the decoder and places it 
# within a larger image with the desired pose
def object_to_image(z_where, obj):
    n = obj.size(0)
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n, 1, 50, 50)))
    out = grid_sample(obj.view(n, 1, 20, 20), grid)
    return out.view(n, 50, 50)

def prior_step(t):
    # Sample object pose. This is a 3-dimensional vector representing 
    # x,y position and size.
    z_where = pyro.sample('z_where_{}'.format(t),
                          dist.normal,
                          z_where_prior_mu,
                          z_where_prior_sigma,
                          batch_size=1)

    # Sample object code. This is a 50-dimensional vector.
    z_what = pyro.sample('z_what_{}'.format(t),
                         dist.normal,
                         z_what_prior_mu,
                         z_what_prior_sigma,
                         batch_size=1)

    # Map code to pixel space using the neural network.
    y_att = decoder(z_what)

    # Position/scale object within larger image.
    y = object_to_image(z_where, y_att)

    return y

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

samples = [prior_step(0)[0] for _ in range(8)]

def show_images(samples):
    plt.rcParams.update({'figure.figsize': [10, 1.6] })
    f, axarr = plt.subplots(1, len(samples))

    for i, img in enumerate(samples):
        axarr[i].imshow(img.data.numpy(), cmap='gray')
        axarr[i].axis('off')

    plt.show()
    
show_images(samples)

def geom_image_prior(x, step=0):
    p = Variable(torch.Tensor([0.4]))
    i = pyro.sample('i{}'.format(step), dist.bernoulli, p)
    if i.data[0] == 1:
        return x
    else:
        # add sampled object to canvas
        x = x + prior_step(step)  
        return geom_image_prior(x, step + 1)

x_empty = ng_zeros(1, 50, 50)
samples = [geom_image_prior(x_empty)[0] for _ in range(16)]
show_images(samples[0:8])
show_images(samples[8:16])

# this is the dataset we used for training
from observations import multi_mnist 
import numpy as np

import pyro.poutine as poutine

from air import AIR, latents_to_tensor
from viz import draw_many, tensor_to_objs, arr2img

def load_data():
    inpath = './multi_mnist_data'
    _, (X_np, Y) = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = Variable(torch.from_numpy(X_np))
    return X

X = load_data()

air = AIR(
    num_steps=3,
    x_size=50,
    z_what_size=50,
    window_size=28,
    encoder_net=[200],
    decoder_net=[200],
    predict_net=[200],
    bl_predict_net=[200],
    rnn_hidden_size=256,
    decoder_output_use_sigmoid=True,
    decoder_output_bias=-2,
    likelihood_sd=0.3
)   

air.load_state_dict(torch.load('air.pyro',
                    map_location={'cuda:0':'cpu'}))

ix = torch.LongTensor([9906, 1879, 5650,  967, 7420, 7240, 2755, 9390,   42, 5584])
n_images = len(ix)
examples_to_viz = X[ix]

params = { 'figure.figsize': [8, 1.6] }   
plt.rcParams.update(params)
f, axarr = plt.subplots(2,n_images)

for i in range(n_images):
    img = arr2img(examples_to_viz[i].data.numpy()).convert('RGB')
    axarr[0,i].imshow(img)
    axarr[0,i].axis('off')

# run the guide and store the sampled random variables in the trace
trace = poutine.trace(air.guide).get_trace(examples_to_viz, None)
# run the prior against the samples in the trace
z, recons = poutine.replay(air.prior, trace)(examples_to_viz.size(0))
# extract the sampled random variables needed to generate the visualization
z_wheres = tensor_to_objs(latents_to_tensor(z))
# make the visualization
recon_imgs = draw_many(recons, z_wheres)
    
for i in range(n_images):
    axarr[1,i].imshow(recon_imgs[i])
    axarr[1,i].axis('off')

plt.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.96, wspace=0.1, hspace=0.1)
plt.savefig('air_multi_mnist_recons.png', dpi=400)
plt.show()

