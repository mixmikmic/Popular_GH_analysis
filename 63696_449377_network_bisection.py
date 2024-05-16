import torch
import torchvision
from torch.autograd import Variable as V
from utils.visualise import make_dot

resnet_18 = torchvision.models.resnet18(pretrained=True)
resnet_18.eval();

# by setting the volatile flag to True, intermediate caches are not saved
# making the inspection of the graph pretty boring / useless
torch.manual_seed(0)
x = V(torch.randn(1, 3, 224, 224))#, volatile=True)
h_x = resnet_18(x)

dot = make_dot(h_x)  # generate network graph
dot.render('net.dot');  # save DOT and PDF in the current directory
# dot  # uncomment for displaying the graph in the notebook

# explore network graph
print('h_x creator ->',h_x.creator)
print('h_x creator prev fun type ->', type(h_x.creator.previous_functions))
print('h_x creator prev fun length ->', len(h_x.creator.previous_functions))
print('\n--- content of h_x creator prev fun ---')
for a, b in enumerate(h_x.creator.previous_functions): print(a, '-->', b)
print('---------------------------------------\n')

print(resnet_18)

resnet_18._modules.keys()

avgpool_layer = resnet_18._modules.get('avgpool')
h = avgpool_layer.register_forward_hook(
        lambda m, i, o: \
        print(
            'm:', type(m),
            '\ni:', type(i),
                '\n   len:', len(i),
                '\n   type:', type(i[0]),
                '\n   data size:', i[0].data.size(),
                '\n   data type:', i[0].data.type(),
            '\no:', type(o),
                '\n   data size:', o.data.size(),
                '\n   data type:', o.data.type(),
        )
)
h_x = resnet_18(x)
h.remove()

my_embedding = torch.zeros(512)
def fun(m, i, o): my_embedding.copy_(o.data)
h = avgpool_layer.register_forward_hook(fun)
h_x = resnet_18(x)
h.remove()

# print first values of the embedding
my_embedding[:10].view(1, -1)

