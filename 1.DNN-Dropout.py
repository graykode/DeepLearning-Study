"""
    Code by Tae Hwan Jung(@graykode)
    Reference : https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/01%20-%20Classification.py
    Dropout example
"""
import torch
import torch.nn as nn

# [fur, wing]
x_data = torch.Tensor([
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = torch.LongTensor([
    0,  # etc
    1,  # mammal
    0,
    0,
    2   # bird
])

new_x_data = torch.Tensor([
    [1, 1],
]) # answer is 2(bird)

criterion = torch.nn.CrossEntropyLoss()

def train(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10000):
        output = model(x_data)

        # when use CrossEntropyLoss, first shape : [batch, n_class], secnod shape : [batch]
        loss = criterion(output, y_data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1)%1000 == 0:
            print("progress:", epoch + 1, "loss=", loss.item())

def test():
    y_pred = model(new_x_data)
    print(y_pred)
    predict = y_pred.max(dim=-1)[1].item()
    print(predict)

class ModelDropout(nn.Module):
    def __init__(self):
        super(ModelDropout, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])
        self.dropout = nn.Dropout(0.5)

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)
        y = self.dropout(y) # make to be comment this line and check diff.
        y = self.w2(y) + self.bias2
        return y

model = ModelDropout()
train(model)
test()