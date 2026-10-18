import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import TenClassNet
import torch

# 0.designated the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1.datasets
train_sets = datasets.MNIST('../data', download=True, train=True)
test_sets = datasets.MNIST('../data', download=True, train=False)

# 2.dataloader
train_loader = DataLoader(train_sets, batch_size=128, shuffle=True)
test_loader = DataLoader(test_sets, batch_size=128, shuffle=True)

# 3.define neural network
m = TenClassNet()
m = m.to(device)

# 4.define loss function
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
criterion.to(device)

# 5.define optimizer
import torch.optim as optim
learning_rate = 1e-2
optimizer = optim.SGD(m.parameters(), lr=learning_rate)

# 6.confirm epoch and training model
epoch = 10
for i in range(epoch):
    total_loss = 0
    for img, lb in train_loader:
        img = img.to(device)
        lb = lb.to(device)
        output = m(img)
        loss = criterion(output, lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch: {i}, loss: {total_loss / len(train_loader):.3f}")

# 7.test the trained model
test_total_loss = 0
m.eval()
with torch.no_grad():
    for img, lb in test_loader:
        img = img.to(device)
        lb = lb.to(device)
        output = m(img)
        loss = criterion(output, lb)
        test_total_loss += loss.item()
    print(f"epoch: {i}, the total loss of test datasets: {test_total_loss / len(test_loader):.3f}")

