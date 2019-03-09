"""
    Code by Tae Hwan Jung(@graykode)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Answer y = x1*1 + x2*2 + x3*3
x_data = Variable(torch.Tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]))
y_data = Variable(torch.Tensor([
    [14.0],
    [32.0],
    [50.0]
]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Model Parameters in Here
        self.w = nn.Linear(3, 1)

    def forward(self, x):
        # feeforwarding of Model in Here
        y = self.w(x)
        return y

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Before training
print("Parameter w :", model.w.weight)

for epoch in range(100):
    output = model(x_data)
    loss = criterion(output, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())

# After training
print("Parameter w :", model.w.weight)