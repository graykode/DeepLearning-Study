"""
    Code by Tae Hwan Jung(@graykode)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable


mnistdata = datasets.MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
print('number of image : ',len(mnistdata))

batch_size = 10
print('Data Loader')
data_loader = torch.utils.data.DataLoader(dataset=mnistdata, batch_size=batch_size, shuffle=True)

count = 0
for batch_idx, (data, target) in enumerate(data_loader):
    data, target = Variable(data), Variable(target)
    count += batch_size
    print('batch :', batch_idx + 1,'    ', count, '/', len(mnistdata),
          'image:', data.shape, 'target : ', target)