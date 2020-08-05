import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # Returns a one-dimensional tensor
'''
arange()和linspace()通过指定开始值、终值创建表示等差数列的一维数组
arange()还需要步长作为参数,得到的结果数组不包含终值。
linspace()还需要元素个数作为参数，可以通过endpoint参数指定是否包含终值，默认为True包含终值。
'''
print(x.size())    # torch.Size([200])
x_np = x.numpy()  # numpy array for plotting
# x_np = x.data.numpy()  # 加data的是过去的写法了,也不会报错.

# following are popular activation functions
y_relu = torch.relu(x).numpy()
y_sigmoid = torch.sigmoid(x).numpy()
y_tanh = torch.tanh(x).numpy()
y_softplus = F.softplus(x).numpy()  # there's no softplus in torch
# y_softmax = torch.softmax(x, dim=0).data.numpy()
# softmax is a special kind of activation function, it is about probability

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
