import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.dim())  # 2
print(x.size())  # 注意:size和size()是有区别的
'''
unsqueeze用于增加维度,第二个参数代表增加维度的地方,维度索引是从0开始算的
索引可以为负数,和python的数组一样,表示的是倒数的意思 
torch.Size([100]) 一维,想要在100后面增加一维变成torch.Size([100,1])则dim=1
torch.Size([100,20]) 二维,
    想要在100前面增加一维变成torch.Size([1,100,20])则dim=0
    想要在100和20中间增加一维变成torch.Size([100,1,20])则dim=1
    想要在20后面增加一维变成torch.Size([100,20,1])则dim=2
对应的squeeze则是把所有长度为1的维度全部去掉

轴(axis): 有多少轴就有多少维度,和维度不等价,轴是有长度的,维度是没有长度的,一个维度有且仅有一个轴
秩(rank): 指轴的数量，或者维度(dimension)的数量，是一个标量
形状(shape): 一个一维数组,元素个数=秩, 每个元素代表一条轴, 数值为轴的长度
'''
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


# torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 用于搭建神经网络
        super(Net, self).__init__()  # 固定操作
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        # Linear(in_features: int, out_features: int, bias: bool = True)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
        '''
        各层之间的输入和输出要相对应 init函数用于构建的神经网络的结构(有多少层,一层有多少神经元)
        self.hidden和self.predict本质上是一个对象(Linear是类,他被实例化了)这些对象只接受一个参数(输入数据)
        这个神经网络必须要继承torch.nn.Module类,在init时要调用父类的构造函数:super(Net, self).__init__()
        同时还必须重写forward函数
        Linear是全连接层/线性层,两个参数(输入数,输出数)
        
        '''

    def forward(self, data):  # 前向传播的方法 data是输入
        data = F.relu(self.hidden(data))  # activation function for hidden layer
        data = self.predict(data)  # linear output
        return data


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 实例化一个优化器
# net.parameters()是优化器要去优化的参数,必须是可迭代的,parameters是Module类的一个成员函数
loss_func = torch.nn.MSELoss()  # 实例化一个损失函数
# this is for regression mean squared error loss

plt.ion()  # something about plotting: Turn the interactive mode on.互动模式

for t in range(200):  # 开始训练
    prediction = net(x)  # input x and predict based on x = data
    # 神经网络接受一个参数: 训练数据  返回预测结果
    loss = loss_func(prediction, y)  # must be (nn output, target)
    # 损失函数接受两个参数: 预测结果和实际结果  返回损失值(应该是张量)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    # Tensor.backward() 计算梯度
    optimizer.step()  # apply gradients
    # 优化器利用梯度优化参数  优化器是怎么知道梯度是多少的?
    # (猜测:optimizer创建的时候关联着net,loss结果应该传回给了net{通过prediction找到网络})
    if t % 5 == 0:  # 画图,每五步更新(画)一次
        # plot and show learning process
        plt.cla()  # Clear the current axes.
        plt.scatter(x.numpy(), y.numpy())  # 绘制散点图
        plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 绘制预测曲线
        # plt.plot(x.numpy(), prediction.detach().numpy(), 'r-', lw=5)   # 绘制预测曲线
        # detach()和data(https://www.cnblogs.com/wanghui-garcia/p/10677071.html)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        # 写出损失率
        plt.pause(0.1)  # Run the GUI event loop for interval(间隔) seconds.

plt.ioff()  # Turn the interactive mode off.
plt.show()
