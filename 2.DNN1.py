"""
    Code by Tae Hwan Jung(@graykode)
    Reference : https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/01%20-%20Classification.py
    1 Layer DNN
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

# [fur, wing]
x_data = torch.Tensor([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = torch.LongTensor([
    0,  # etc
    1,  # mammal
    2,  # birds
    0,
    0,
    2
])

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.w = nn.Linear(2, 3)
        self.bias = torch.zeros([3])
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.w(x) + self.bias
        y = self.relu(y)
        return y

model = DNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    output = model(x_data)

    # when use CrossEntropyLoss, first shape : [batch, n_class], secnod shape : [batch]
    loss = criterion(output, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())

# test
for x in x_data:
    y_pred = model(x)
    print(y_pred)
    predict = y_pred.max(dim=0)[1].item()
    print(predict)