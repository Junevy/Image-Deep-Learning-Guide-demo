import torch.nn as nn
import torch

class CustomMaxPool(nn.Module):
    def __init__(self):
        super(CustomMaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.maxpool(x)
        return x

input = torch.tensor(
    [
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.],
        [26., 27., 28., 29., 30.],
        [31., 32., 33., 34., 35.],
    ])

# input = input.reshape(-1, 1, 7, 5)
# print(input.shape)
# print(input)

pool = CustomMaxPool()
output = pool(input)
print(output.shape)
print(output)