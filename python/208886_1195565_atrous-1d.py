import numpy as np

x = np.zeros((3))
rate = 2
x

atrous = np.ones(np.array(x.shape) + rate)
atrous

for i in range(0, atrous.shape[0], rate //2+1):
    atrous[i] = atrous[i] * x[int(i/rate/2)+1]
atrous

x = np.random.rand(1,7,3)
kernel = np.random.rand(3,3,7)
filter_size = kernel.shape[0]
stride = 2
rate = 2

def padding(x, filter_size, pad='SAME'):
    if pad == 'SAME':
        pad_h_min = int(np.floor((filter_size - 1)/2))
        pad_h_max = int(np.ceil((filter_size - 1)/2))
        pad_h = (pad_h_min, pad_h_max)
        return np.pad(x, ((0, 0), pad_h, (0, 0)), mode='constant')
    else:
        return x
    
def get_shape(x):
    output_height = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
    return int(output_height)

x_padded = padding(x, filter_size)
h = get_shape(x_padded)
out_atrous = np.zeros((1, h, kernel.shape[2]))
out_atrous.shape

def atrous(x, w):
    for i in range(0, x.shape[0], rate //2+1):
        x[i,:] = x[i,:] * w[int(i/rate/2)+1,:]
    return x

def conv(x, w, out):
    for k in range(x.shape[0]):
        for z in range(w.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                atroused = atrous(x[k, _h * stride:_h * stride + filter_size + rate, :], w[:, :, z])
                out[k, _h, z] = np.sum(atroused)
    return out

out_atrous = conv(x_padded, kernel, out_atrous)
out_atrous.shape

def deatrous_w(x, w, de):
    for i in range(0, x.shape[0], rate //2+1):
        w[int(i/rate/2)+1,:] = np.sum(x[i,:] * de[i,:])
    return w

def deconv_w(x, w, de):
    for k in range(x.shape[0]):
        for z in range(w.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                weighted = deatrous_w(x[k, _h * stride:_h * stride + filter_size + rate, :], w[:, :, z],
                                         de[k, _h * stride:_h * stride + filter_size + rate, :])
                w[:, :, z] = weighted
    return w

def deconv_x(x, w, de):
    for k in range(x.shape[0]):
        for z in range(x.shape[2]):
            h_range = int(np.ceil((x.shape[1] - rate * (filter_size-1)) / stride) + 1)
            for _h in range(h_range):
                atroused = atrous(de[k, _h * stride:_h * stride + filter_size + rate, :], w[:, z, :])
                x[k, _h, z] = np.sum(atroused)
    return x

dkernel = np.zeros(kernel.shape)
deconv_w(out_atrous, dkernel, out_atrous).shape

dx = np.zeros(x.shape)
deconv_x(dx, kernel, out_atrous).shape



