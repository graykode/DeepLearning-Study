"""
    Code by Tae Hwan Jung(@graykode)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

train_data = datasets.MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
train_batch_size = 5000
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

    def forward(self, x):
        """
        put your code
        """
        out = x
        return out

model = DNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training
count = 0
model.train()
for batch_idx, (data, target) in enumerate(train_dataloader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    count += train_batch_size
    print('batch :', batch_idx + 1,'    ', count, '/', len(train_data),
          'image:', data.shape, 'target : ', target)

# Test
test_data = datasets.MNIST('./data/', train=False, download=True, transform=transforms.ToTensor())
test_batch_size = 5000
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=test_batch_size, shuffle=True)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_dataloader:
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_dataloader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_dataloader.dataset),
    100. * correct / len(test_dataloader.dataset)))