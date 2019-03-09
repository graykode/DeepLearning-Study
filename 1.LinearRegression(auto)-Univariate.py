"""
    Code by Tae Hwan Jung(@graykode)
    Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/05_linear_regression.py
"""
import torch
import torch.nn as nn
import torch.optim as optim

x_data = torch.Tensor([
    [1.0],
    [2.0],
    [3.0]
])
y_data = torch.Tensor([
    [2.0],
    [4.0],
    [6.0]
])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Model Parameters in Here
        self.w = nn.Linear(1, 1)

    def forward(self, x):
        # feeforwarding of Model in Here
        y = self.w(x)
        return y

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Before training
print("Parameter w :", model.w.weight.item(), "Predict y :" , model(torch.Tensor([4.0])).item() )

for epoch in range(10):
    output = model(x_data)
    loss = criterion(output, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())

# After training
print("Parameter w :", model.w.weight.item(), "Predict y :" , model(torch.Tensor([4.0])).item() )