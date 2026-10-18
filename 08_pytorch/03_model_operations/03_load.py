import torch
import torchvision

# load method 1
model = torch.load('vgg16_no_pretrained.pth', weights_only=False)
print(model)

# load method 2
dict = torch.load('vgg16_no_pretrained_dict.pth')
# print(dict)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(dict)
print(vgg16)
