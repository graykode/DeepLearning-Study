"""
    Code by Tae Hwan Jung(@graykode)
    Reference : https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/01%20-%20Classification.py
    2 Layer DNN
"""
import torch
import torch.nn as nn

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
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)

        y = self.w2(y) + self.bias2
        return y

model = DNN()

criterion = torch.nn.CrossEntropyLoss()

# See http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
# Pytorch https://pytorch.org/docs/stable/optim.html
optimSGD = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
optimAdam = torch.optim.Adam(model.parameters(), lr=0.01)
optimAdagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
optimAdadelta = torch.optim.Adadelta(model.parameters(), lr=0.01)
optimRMSprop = torch.optim.RMSprop(model.parameters(), lr=0.01)

for epoch in range(1000):
    output = model(x_data)

    # when use CrossEntropyLoss, first shape : [batch, n_class], secnod shape : [batch]
    loss = criterion(output, y_data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimSGD.zero_grad()
    loss.backward()
    optimSGD.step()

    print("progress:", epoch, "loss=", loss.item())

# test
for x in x_data:
    y_pred = model(x)
    print(y_pred)
    predict = y_pred.max(dim=0)[1].item()
    print(predict)