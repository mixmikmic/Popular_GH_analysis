get_ipython().magic('pylab inline')
figsize(10,5)
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams["image.cmap"] = "afmhot"

import clstm
import h5py

get_ipython().system('test -f mnist_seq.h5 || curl http://www.tmbdev.net/ocrdata-hdf5/mnist_seq.h5 > mnist_seq.h5 || rm -f mnist_seq.h5')

h5 = h5py.File("mnist_seq.h5","r")
imshow(h5["images"][0].reshape(*h5["images_dims"][0]))
print h5["images"].shape

net = clstm.make_net_init("bidi","ninput=28:nhidden=10:noutput=11")
net.setLearningRate(1e-2,0.9)
print clstm.network_info(net)

print [chr(c) for c in h5["codec"]]

index = 0
xs = array(h5["images"][index].reshape(28,28,1),'f')
cls = h5["transcripts"][index][0]
print cls
imshow(xs.reshape(28,28).T,cmap=cm.gray)

net.inputs.aset(xs)
net.forward()
pred = net.outputs.array()
imshow(pred.reshape(28,11).T, interpolation='none')

target = zeros((3,11),'f')
target[0,0] = 1
target[2,0] = 1
target[1,cls] = 1
seq = clstm.Sequence()
seq.aset(target.reshape(3,11,1))
aligned = clstm.Sequence()
clstm.seq_ctc_align(aligned,net.outputs,seq)
aligned = aligned.array()
imshow(aligned.reshape(28,11).T, interpolation='none')

deltas = aligned - net.outputs.array()
net.d_outputs.aset(deltas)
net.backward()
net.update()

for i in range(60000):
    index = int(rand()*60000)
    xs = array(h5["images"][index].reshape(28,28,1),'f')
    cls = h5["transcripts"][index][0]
    net.inputs.aset(xs)
    net.forward()
    pred = net.outputs.array()
    target = zeros((3,11),'f')
    target[0,0] = 1
    target[2,0] = 1
    target[1,cls] = 1
    seq = clstm.Sequence()
    seq.aset(target.reshape(3,11,1))
    aligned = clstm.Sequence()
    clstm.seq_ctc_align(aligned,net.outputs,seq)
    aligned = aligned.array()
    deltas = aligned - net.outputs.array()
    net.d_outputs.aset(deltas)
    net.backward()
    net.update()

figsize(5,10)
subplot(211,aspect=1)
imshow(xs.reshape(28,28).T)
subplot(212,aspect=1)
imshow(pred.reshape(28,11).T, interpolation='none', vmin=0, vmax=1)

