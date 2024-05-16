get_ipython().magic('pylab inline')
figsize(10,5)

import clstm

net = clstm.make_net_init("lstm1","ninput=1:nhidden=4:noutput=2")
print net

net.setLearningRate(1e-4,0.9)
print clstm.network_info_as_string(net)

print net.sub.size()
print net.sub[0]
print net.sub[0].kind

N = 20
xs = array(randn(N,1,1)<0.2, 'f')
net.inputs.aset(xs)
net.forward()

N = 20
test = array(rand(N)<0.3, 'f')
plot(test, '--', c="black")
ntrain = 30000
for i in range(ntrain):
    xs = array(rand(N)<0.3, 'f')
    ys = roll(xs, 1)
    ys[0] = 0
    ys = array([1-ys, ys],'f').T.copy()
    net.inputs.aset(xs.reshape(N,1,1))
    net.forward()
    net.outputs.dset(ys.reshape(N,2,1)-net.outputs.array())
    net.backward()
    clstm.sgd_update(net)
    if i%1000==0:
        net.inputs.aset(test.reshape(N,1,1))
        net.forward()
        plot(net.outputs.array()[:,1,0],c=cm.jet(i*1.0/ntrain))



