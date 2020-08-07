# standard library
import os  # python标准库之一,用于文件操作
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

# The torchvision package consists of popular datasets, model architectures,
# and common image transformations for computer vision.

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
# 实际上1也做够好了(0.98),此时加再多epoch也不会有太多增长(我试了5 epoch,后面也都是98/99的变)
BATCH_SIZE = 50
LR = 0.001  # learning rate
DOWNLOAD_MNIST = False

# Mnist digits dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(  # 使用torchvision载入MNIST数据集
    root='./mnist/',
    train=True,  # If True, creates dataset from training.pt, otherwise from test.pt.
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST  # 是否下载MNIST数据集
)

# plot one example
print(train_data.train_data.size())  # (60000, 28, 28)
print(train_data.train_labels.size())  # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(  # 分批训练
    dataset=train_data,  # 导入的MNIST数据集
    batch_size=BATCH_SIZE,  # 分批
    shuffle=True  # 打乱
)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.
# .type(torch.FloatTensor)用来将test_data所有元素转换为FloatTensor类型
# .unsqueeze(test_data.test_data, dim=1)用来升维,变成秩为2的数据
# [:2000]用来只取前2000的数据
# /255. 将数组的所有元素除以255.(浮点数) 归一化所有像素值

# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):  # 构建模型
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 因为有padding所以卷积过后图片不会变小
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)  (通道,高,宽)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes
        # 7 = (28/2)/2 两次池化导致图片变小,全连接层每个单元都要连接上一层所有单元,通道数X高X宽

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # 将图片(三维)拉成一维            (batch_size, 32 , 7 , 7) -> (batch_size, 32 * 7 * 7)
        _output = self.out(x)
        return _output, x  # return x for visualization 返回x为了可视化
        # 注意这里又两个返回值,下面调用的时候应该用两个变量来接收,否则加一个索引
        # 例如: test_output, last_layer = cnn(test_x)  又例如: output = cnn(b_x)[0]


cnn = CNN()  # 实例化
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
# following function (plot_with_labels) is for visualization, can be ignored if not interested


try:
    from sklearn.manifold import TSNE

    '''
    scikit
    manifold learning流形学习是一种非线性降维的手段。多维度数据集非常难于可视化。
    反而2维或者3维数据很容易通过图表展示数据本身的内部结构，等价的高维绘图就远没有那么直观了。
    为了实现数据集结构的可视化，数据的维度必须通过某种方式降维。
    '''

    HAS_SK = True
# except IndexError:
except ImportError:     # Expection是所有异常的基类,不具体,具体的异常类有哪些可以看最下面
    HAS_SK = False
    print('Please install sklearn for layer visualization')


# import matplotlib

def plot_with_labels(lowDWeights, _labels):
    plt.cla()  # Clear the current axes.
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, _labels):
        # c = matplotlib.cm.rainbow(int(255 * s / 9))   # 按理来说是这样的
        # 文档找不到rainbow,大致用来按照数值取颜色(把255分成9份)
        # 不知道为什么文档里面找不到pyplot.cm内容但是可以引用到,如下
        c = plt.cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())  # 限制图像宽度在X的取值范围
    plt.ylim(Y.min(), Y.max())  # 限制图像高度在Y的取值范围
    plt.title('Visualize last layer')  # 图表的标题
    plt.show()
    plt.pause(0.01)


plt.ion()
# training and testing
for epoch in range(EPOCH):  # 训练
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        # enumerate在遍历的时候自动生成序数在前面,这里我们用step来接收序数,而用(b_x,b_y)来接收train_loader的值
        output = cnn(b_x)[0]  # cnn output # cnn的forward函数有两个返回值 output和x ,加一个索引可以只取output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients 注意这个step()是优化器的函数而不是上面的序数(下一行的step才是序数)

        if step % 50 == 0:  # 没训练50次测试一次,并计算正确率
            test_output, last_layer = cnn(test_x)
            '''  cnn神经网络中定义的forward函数就是实例变量cnn(test_x)的返回值,
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = x.view(x.size(0), -1)                
                _output = self.out(x)
                return _output, x
            '''
            pred_y = torch.max(test_output, 1)[1].data.numpy()    # torch.max()有几个重载,这是其中一个,获取值以及索引
            # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
            # 返回一个命名元组(值，索引)，其中值是输入张量在给定维数dim中每一行的最大值。索引是找到的每个最大值的索引位置(argmax)。
            # 下面注释过长,建议通读源码时折叠
            '''
            torch.Size([2000, 10])
            test_output 是一个二维数组,神经网络的输出都是二维数组,数组大小为data_num * output_classes_num,
            在这个神经网络我们规定所有数据只有2000个,因为该网络用于识别数字,两千个数据每个数据进入神经网络都有十个输出
            
            对于任意维的张量 arr[][]....[] dim维度从前到后 dim为从0到n,也就是说dim=0永远是最外围,dim最大的时候取到的是数字.
            dim对应 哪一个轴 就沿着 哪一个轴 进行若干次{操作}.
            二维数组中,dim=0为y轴,dim=1为x轴,
            [[1,2,3], 
             [3,4,5]]
            对以上数组进行{求最大值操作} 当dim=0时,结果为[3,4,5] 当dim=1时,结果为[3,5]   
            三维数组中,dim=0为z轴,dim=1为y轴,dim=2为x轴
            [[[1,2,3,9], 
              [3,4,5,7]],
              
             [[2,3,4,5],
              [1,2,3,4]],
        
             [[2,3,4,5],
              [1,2,3,4]]]
            对以上数组进行{求最大值操作} 
            当dim=0时,结果为[[2,3,4,9],[3,4,5,7]] 
            当dim=1时,结果为[[3,4,5,9],[2,3,4,5],[2,3,4,5]] 
            当dim=2时,结果为[[9,7],[5,4],[5,4]]
            
            沿着dim=n的方向取最大值得到形状为除去原数组形状索引为n的值的形状,
            当数组形状为(2,3) 沿dim=0求最大值,结果的形状为出去索引为0的值(此处为2)的形状,所以得到的结果的形状为(3)
            当数组形状为(2,3) 沿dim=1求最大值,结果的形状为出去索引为0的值(此处为3)的形状,所以得到的结果的形状为(2)
            当数组形状为(3,2,4) 沿dim=0求最大值,结果的形状为出去索引为0的值(此处为3)的形状,所以得到的结果的形状为(2,4)
            当数组形状为(3,2,4) 沿dim=1求最大值,结果的形状为出去索引为0的值(此处为3)的形状,所以得到的结果的形状为(3,4)
            当数组形状为(3,2,4) 沿dim=2求最大值,结果的形状为出去索引为0的值(此处为3)的形状,所以得到的结果的形状为(2,3)
            '''
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            # 计算正确率/置信度 相等结果为True,用astype转化为int再求和得到的结果和总数相除计算得出
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:  # 如果sklearn可以用的话就将数据降维
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)     # 定义一个降维器
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                # fit_transform(X, y=None)  Fit X into an embedded space and return that transformed output.
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)  # 调用上面定义的画图函数
                '''
                TSNE(
                     n_components=2,   # Dimension of the embedded space.
                     perplexity=30.0,   # related to the number of nearest neighbors 
                     init='random',   # Initialization of embedding.options:‘random’, ‘pca’, and a numpy array of shape
                     n_iter=1000,   # Maximum number of iterations for the optimization. Should be at least 250.
                     n_iter_without_progress=300,   # Maximum number of iterations without progress
                     min_grad_norm=1e-07,  # optimization will be stopped when gradient norm is below this threshold
                     early_exaggeration=12.0, 
                     learning_rate=200.0, 
                     metric='euclidean', 
                     verbose=0, 
                     random_state=None, 
                     method='barnes_hut', 
                     angle=0.5, 
                     n_jobs=None
                 )
                '''
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

'''
BaseException  所有异常的基类
     |
     +-- SystemExit  解释器请求退出
     |
     +-- KeyboardInterrupt  用户中断执行(通常是输入^C)
     |
     +-- GeneratorExit  生成器调用close（）方法时触发的
     |
     +-- Exception  常规错误的基类，异常都是从基类Exception继承的。
          |
          +-- StopIteration  迭代器没有更多的值
          |
          +-- StandardError  所有的内建标准异常的基类
          |    |
          |    +-- BufferError  缓冲区操作不能执行
          |    |
          |    +-- ArithmeticError  所有数值计算错误的基类
          |    |    |    
          |    |    +-- FloatingPointError  浮点计算错误
          |    |    |    
          |    |    +-- OverflowError  数值运算超出最大限制
          |    |    |        
          |    |    +-- ZeroDivisionError  除(或取模)零 (所有数据类型)
          |    |
          |    +-- AssertionError  断言语句失败
          |    |
          |    +-- AttributeError  访问未知对象属性
          |    |
          |    +-- EnvironmentError  操作系统错误的基类
          |    |    +-- IOError  输入输出错误
          |    |    |    
          |    |    +-- OSError  操作系统错误
          |    |         |
          |    |         +-- WindowsError (Windows)  系统调用失败
          |    |         |    
          |    |         +-- VMSError (VMS)  系统调用失败
          |    |
          |    +-- EOFError  没有内建输入,到达EOF 标记
          |    |
          |    +-- ImportError  导入模块/对象失败
          |    |
          |    +-- LookupError  无效数据查询的基类，键、值不存在引发的异常
          |    |    |    
          |    |    +-- IndexError  索引超出范围
          |    |    |    
          |    |    +-- KeyError  字典关键字不存在
          |    |
          |    +-- MemoryError  内存溢出错误(对于Python 解释器不是致命的)
          |    |
          |    +-- NameError  未声明/初始化对象 (没有属性)
          |    |    |    
          |    |    +-- UnboundLocalError  访问未初始化的本地变量
          |    |
          |    +-- ReferenceError  弱引用(Weak reference)试图访问已经垃圾回收了的对象
          |    |
          |    +-- RuntimeError  一般的运行时错误
          |    |    |
          |    |    +-- NotImplementedError  尚未实现的方法
          |    |
          |    +-- SyntaxError  语法错误
          |    |    |
          |    |    +-- IndentationError  缩进错误
          |    |         |
          |    |         +-- TabError  Tab和空格混用
          |    |
          |    +-- SystemError  一般的解释器系统错误
          |    |
          |    +-- TypeError  对类型无效的操作
          |    |
          |    +-- ValueError  传入无效的参数
          |         +-- UnicodeError  Unicode 相关的错误
          |              |
          |              +-- UnicodeDecodeError  Unicode 解码时的错误
          |              |    
          |              +-- UnicodeEncodeError  Unicode 编码时错误
          |              |
          |              +-- UnicodeTranslateError  Unicode 转换时错误
          |
          +-- Warning  警告的基类
               |
               +-- DeprecationWarning  关于被弃用的特征的警告
               |
               +-- PendingDeprecationWarning  关于特性将会被废弃的警告
               |
               +-- RuntimeWarning  可疑的运行时行为(runtime behavior)的警告
               |
               +-- SyntaxWarning  可疑的语法的警告
               |
               +-- UserWarning  用户代码生成的警告
               |
               +-- FutureWarning  关于构造将来语义会有改变的警告
               |
               +-- ImportWarning  关于模块进口可能出现错误的警告的基类。
               |
               +-- UnicodeWarning  有关Unicode警告的基类。
               |
               +-- BytesWarning  有关字节警告相关的基类。
'''
