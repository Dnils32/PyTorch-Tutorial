import torch
import torch.nn.functional as F
from collections import OrderedDict     # python标准库之一


# replace following class code with an easy sequential network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net1 = Net(1, 10, 1)  # 以上是传统的方法搭建一个神经网络

# 下面是更快的方法,利用torch.nn里面的Sequential容器(Module是容器之一)
# easy and fast way to build your network
net2 = torch.nn.Sequential(  # 神经网络会按照顺序来传递数据
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

net3 = torch.nn.Sequential(OrderedDict([    # 使用OrderedDict要倒入collections
    ('hidden', torch.nn.Linear(1, 10)),
    ('relu1', torch.nn.ReLU()),
    ('predict', torch.nn.Linear(10, 1)),
]))

print(net1)  # net1 architecture
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

print(net2)  # net2 architecture
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""
print(net3)  # net3 architecture
'''
Sequential(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (relu1): ReLU()
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
'''
# 三个网络输出还是有一些不一样的,sequential有两种写法
