"""
    Code by Tae Hwan Jung(@graykode)
"""
import torch
import torch.nn as nn
import torch.optim as optim

x_data = torch.Tensor([
    [-2.0],
    [-1.0],
    [1.0],
    [5.0]
])

y_data = torch.LongTensor([1, 1, 2, 3]) # 3 classes

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Model Parameters in Here
        self.w = nn.Linear(1, 4)

    def forward(self, x):
        # feeforwarding of Model in Here
        y = self.w(x)
        return y

model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
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