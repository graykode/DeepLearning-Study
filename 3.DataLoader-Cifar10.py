"""
    Code by Tae Hwan Jung(@graykode)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

mnistdata = datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.ToTensor())
print('number of image : ',len(mnistdata))

batch_size = 10
print('Data Loader')
data_loader = torch.utils.data.DataLoader(dataset=mnistdata, batch_size=batch_size, shuffle=True)

count = 0
for batch_idx, (data, targets) in enumerate(data_loader):
    targets = [classes[target] for target in targets]
    data = Variable(data)
    count += batch_size
    print('batch :', batch_idx + 1,'    ', count, '/', len(mnistdata),
          'image:', data.shape, 'target : ', targets)