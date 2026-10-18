import torchvision
import torch

# save method 1
vgg16_no_trained = torchvision.models.vgg16(pretrained=False)
torch.save(vgg16_no_trained, 'vgg16_no_pretrained.pth')

# save method 2 ( recommand )
torch.save(vgg16_no_trained.state_dict(), 'vgg16_no_pretrained_dict.pth')
