import torch.nn as nn
import torch

class TenClassNet(nn.Module):
    def __init__(self):
        super(TenClassNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    m = TenClassNet()
    input = torch.ones((64, 1, 28, 28))
    output = m(input)
    print(output.size())