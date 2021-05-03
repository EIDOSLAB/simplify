import torch
import torchvision
import torch.nn.utils.prune as prune
import simplify

import time

if __name__ == '__main__':
    device = 'cpu'

    model = torchvision.models.vgg11(pretrained=False).to(device)
    x = torch.randn((10, 3, 224, 224)).to(device)

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        y = model(x)
        end = time.perf_counter()

    print('Took', end-start, 'seconds')

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')

    with torch.no_grad():
        start = time.perf_counter()
        y = model(x)
        end = time.perf_counter()

    print('Took', end-start, 'seconds')