import time

import torch
import torch.nn.utils.prune as prune
from EIDOSearch.pruning.simplification import simplify
# import simplify
from torchvision.models import resnet18

if __name__ == '__main__':
    device = 'cpu'
    
    model = resnet18().to(device)
    x = torch.randn((10, 3, 224, 224)).to(device)
    
    model.eval()
    with torch.no_grad():
        model(x)  # warmup
        start = time.perf_counter()
        y = model(x)
        end = time.perf_counter()
    
    print('Took', end - start, 'seconds')
    
    for name, module in model.named_modules():
        if name == "fc2":
            break
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.9, dim=0)
            prune.remove(module, 'weight')
    
    with torch.no_grad():
        start = time.perf_counter()
        y = model(x)
        end = time.perf_counter()
        print(torch.sum(torch.abs(y)))
    
    print('Took', end - start, 'seconds')
    
    model = simplify(model)
    
    with torch.no_grad():
        start = time.perf_counter()
        y = model(x)
        end = time.perf_counter()
        print(torch.sum(torch.abs(y)))
    
    print('Took', end - start, 'seconds')
