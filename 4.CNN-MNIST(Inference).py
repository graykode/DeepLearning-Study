'''
    code by Tae Hwan Jung @graykode
    reference : https://github.com/pytorch/examples/blob/master/mnist/main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4 * 4 * 50) # [batch_size, 50, 4, 4]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def img_show(image):
    plt.imshow(image.numpy().reshape((28, 28)), cmap='gray')
    plt.show(block=False)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_data = datasets.MNIST('./data', train=False, transform=transform)

    model = Model().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt', map_location=lambda storage, loc: storage))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    with torch.no_grad():
        for i in range(10):
            data = test_data[i][0]
            target = test_data[i][1]
            img_show(data)

            data = data.unsqueeze(0)
            data = data.to(device)

            output = model(data)
            predict = output.max(dim=1)[1]
            print('model predict : ', predict, ' answer : ',target)