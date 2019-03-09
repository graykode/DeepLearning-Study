"""
    Code by Tae Hwan Jung(@graykode)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

x_data = Variable(torch.Tensor([
    [-2.0],
    [-1.0],
    [1.0],
    [5.0]
]))

y_data = Variable(torch.Tensor([
    [1.0],
    [1.0],
    [2.0],
    [2.0]
]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Model Parameters in Here
        self.w = nn.Linear(1, 1)

    def forward(self, x):
        # feeforwarding of Model in Here
        y = self.w(x)
        y = torch.sigmoid(y)
        return y

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    output = model(x_data)
    loss = criterion(output, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())

# test
for x in x_data:
    y_pred = torch.sigmoid(x)
    print(y_pred)
    print(1 if y_pred >= 0.5 else 0)