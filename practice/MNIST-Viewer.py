"""
    Code by Tae Hwan Jung(@graykode)
"""
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def show(image):
    pixels = np.array(image, dtype='float').reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

mnistdata = datasets.MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
# first is index, second is (image, label)
print(mnistdata[0][1])
show(mnistdata[0][0])