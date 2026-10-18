import torch
import torch.nn as nn

inputs = torch.tensor(
    [
        [-1., 2., 3., 4., 5.],
        [6., -7., 8., 9., 10.],
        [11., 12., -13., 14., 15.],
        [16., 17., 18., -19., 20.],
        [21., 22., 23., 24., -25.],
        [26., 27., 28., -29., 30.],
        [31., 32., -33., 34., 35.],
    ])

class NonLinearRelu(nn.Module):
    def __init__(self):
        super(NonLinearRelu, self).__init__()
        self.relu = nn.ReLU(inplace=True)       # inplace: 是否修改输入张量？

    def forward(self, x):
        x = self.relu(x)
        return x

linear = NonLinearRelu()
print(inputs.shape)

# discard the outputs(the value of return) when the inplace is true.
linear(inputs)
print(inputs.shape)
print(inputs)