from typing import List
import torch
import numpy as np

# details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))  # reshape 重塑 把1X6矩阵变为2X3矩阵
# numpy.arange([start=0, ]stop, [step=1, ]dtype=None) np.arange(6) ->[0 1 2 3 4 5]
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,  # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,  # 0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array,  # [[0 1 2], [3 4 5]]
)

# abs
data: List[int] = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),  # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)  # [1 2 1 2]
)

# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),  # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),  # 0.0
    '\ntorch: ', torch.mean(tensor)  # 0.0
)

# matrix multiplication
data2 = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data2)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data2, data2),  # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)  # [[7, 10], [15, 22]]
)

'''
点乘是对应位置元素相乘，要求两矩阵必须尺寸相同；
叉乘是矩阵a的第一行乘以矩阵b的第一列，各个元素对应相乘然后求和作为第一元素的值，
要求矩阵a的列数等于矩阵b的行数，乘积矩阵的行数等于左边矩阵的行数,乘积矩阵的列数等于右边矩阵的列数。
所以正确的说法应该是：
numpy.matmul() 和torch.mm() 是矩阵乘法（叉乘），
numpy.multiply() 和 torch.mul() 是矩阵点乘（对应元素相乘）
'''

# incorrect method
# data2 = np.array(data2)
# print(
#     '\nmatrix multiplication (dot)',
#     '\nnumpy: ', data.dot(data2),  # [[7, 10], [15, 22]]
#     '\ntorch: ', tensor.dot(tensor)  # this will convert tensor to [1,2,3,4], you'll get 30.0
# )
