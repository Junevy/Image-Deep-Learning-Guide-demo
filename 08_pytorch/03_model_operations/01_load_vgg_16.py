import torch.nn as nn
import torchvision

vgg16_no_pretrained = torchvision.models.vgg16(pretrained=False)
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)
print(vgg16_no_pretrained)

# custom linear layer
# method 1
# vgg16_pretrained.classifier.add_module('custom_linear', nn.Linear(1000, 10))
# print(vgg16_pretrained)

# method 2
vgg16_pretrained.classifier[6] = nn.Linear(4096, 10)
print(vgg16_pretrained)
