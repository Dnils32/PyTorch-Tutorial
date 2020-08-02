import torch
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


# 要拟合出一条x^2的函数,用rand制造一些随机但是像x^2的数据,tensor之间相加size要一样,所以rand的大小为x.size

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():  # 保存神经网络
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # global prediction  #  全局变量为了是prediction在下面画图的时候不至于未定义
    # 全局变量会导致下面重名,所以改成这样更好(也就是说使用全局变量有些多余)
    prediction = net1()
    for t in range(100):  # 训练
        prediction = net1(x)  # 注意这里是第一次出现prediction,但是作用域仅在这个循环里面,应该用global声明一下
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result 画图
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    # 子图,一张图显示多个图像,这个是其中一个,此处参数缩写了,意思是将一个图分成1行3列,这个子图放在第1列 简写131
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    # Local variable 'prediction' might be referenced before assignment 要全局声明否则报错!

    # 2 ways to save the net
    torch.save(net1, 'net.pkl')  # save entire net
    torch.save(net1.state_dict(), 'net_params.pkl')  # save only the parameters
    # 第二种方法更加好 state_dict() 是Module类的成员函数,状态字典


def restore_net():  # 读取整个网络
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')    # load函数用于下载网络模型文件,输入URL
    # prediction2 = net2(x)   # 前面prediction被全局声明这里要改名 # Shadows name 'prediction' from outer scope

    prediction = net2(x)

    # plot result
    plt.subplot(132)    # 参数缩写了,意思是将一个图分成1行3列,这个子图放在第2列 简写132
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():   # 读取网络参数
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(     # 因为当时只存了参数,所以要先构建一个和之前一样的模型
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))   # 先读取网络模型文件,再将参数赋值给新构建的模型
    # prediction3 = net3(x)   # 前面prediction被全局声明这里要改名 # Shadows name 'prediction' from outer scope
    prediction = net3(x)

    # plot result
    plt.subplot(133)    # 参数缩写了,意思是将一个图分成1行3列,这个子图放在第3列 简写133
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()
