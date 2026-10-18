import torch.nn as nn
from torchvision import datasets
import torch.utils.data as data
from torchvision.transforms import transforms

class SimpleCnnNet(nn.Module):
    def __init__(self):
        super(SimpleCnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

tr_sets = datasets.MNIST(root='/data', download=True, transform=transforms.ToTensor())
tr_loader = data.DataLoader(tr_sets, batch_size=64, shuffle=True)
cnn_net = SimpleCnnNet()

for img, label in tr_loader:
    print(img.shape)
    out_img = cnn_net(img)
    print(out_img.shape)




