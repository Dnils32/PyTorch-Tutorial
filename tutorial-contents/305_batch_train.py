import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 设置批的大小

# 做一些数据
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
torch_dataset = Data.TensorDataset(x, y)     # 将输入和输出加入到数据集中
# torch.utils.data.TensorDataset(*tensors)  Dataset wrapping tensors.
# Each sample will be retrieved by indexing tensors along the first dimension.
loader = Data.DataLoader(   # 负载,加载器,一个loader就是内部被切分成若干批的数据集
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    # 随机打乱(批梯度下降最好要随机打乱保证从数据集采样比较均匀)
    num_workers=2,              # subprocesses for loading data
    # 同时工作的worker的数量
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
    '''
    一个python文件通常有两种使用方法，第一是作为脚本直接执行，
    第二是 import 到其他的 python 脚本中被调用（模块重用）执行。
    因此 if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
    在 if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行
    而 import 到其他脚本中是不会被执行的
    每个python模块都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名.py
    如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）。
    “__main__” 始终指当前执行模块的名称.py。进而当模块被直接执行时，__name__ == 'main' 结果为真。
    '''
    print(__name__)
