"""
    Code by Tae Hwan Jung(@graykode)
"""
import torch.nn as nn

class Model(nn.Module):
    def __init__(self , W , b):
        super(Model, self).__init__()
        # Model Parameters in Here
        self.W = W
        self.b = b

    def forward(self, x):
        # feeforwarding of Model in Here
        y = self.W * x + self.b
        return y

model = Model(W=10, b=5)
output = model(1) # 10 * 1 + 5
print(output)

