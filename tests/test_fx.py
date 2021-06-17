import torch
from torch.fx._experimental.fuser import fuse
from torchvision.models import resnet18, densenet121, googlenet

if __name__ == '__main__':
    x = torch.rand(10, 3, 224, 224)
    
    model = googlenet(True)
    model.eval()
    y_src = model(x)
    
    model = fuse(model)
    model.eval()
    y_prop = model(x)
    print(model)
