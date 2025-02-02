import torch
from torch import nn
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.01  # learning rate
DOWNLOAD_MNIST = True  # set to True if haven't download the data

# Mnist digital dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)
# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor()
)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255.  # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]  # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension.
            # e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, c_n) = self.rnn(x, None)  # None represents zero initial hidden state
        '''  self.rnn = nn.LSTM   
        Inputs: input, (h_0, c_0)
        input : tensor containing the features of the input sequence.         
                shape (seq_len, batch, input_size)
        h_0 : tensor containing the initial hidden state for each element in the batch.
                shape (num_layers * num_directions, batch, hidden_size)
        c_0 : tensor containing the initial cell state for each element in the batch.
                shape (num_layers * num_directions, batch, hidden_size)
        
        Outputs: output, (h_n, c_n)
        output : tensor containing the output features (h_t) from the last layer of the LSTM
                shape (seq_len, batch, num_directions * hidden_size)
        h_n : tensor containing the hidden state for t = seq_len.
                shape (num_layers * num_directions, batch, hidden_size)
        c_n : tensor containing the cell state for t = seq_len.
                shape (num_layers * num_directions, batch, hidden_size)
        '''
        out = self.out(r_out[:, -1, :])   # choose r_out at the last time step
        return out


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # 10个值中最大的值为最可能的值,训练时,我们将索引和答案标签对应起来
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
