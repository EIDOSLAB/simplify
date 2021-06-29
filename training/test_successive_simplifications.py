import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.models import resnet18

import utils
from simplify import propagate_bias, remove_zeroed, fuse

if __name__ == '__main__':
    utils.set_seed(3)
    model = resnet18(True)
    
    bn_folding = utils.get_bn_folding(model)
    model = fuse(model, bn_folding)
    model.eval()
    pinned_out = utils.get_pinned_out(model)
    
    im = torch.randint(0, 256, (10, 3, 224, 224))
    x = im / 255.
    
    zeros = torch.zeros(1, *im.shape[1:])
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, name="weight", amount=0.5, dim=0)
            prune.remove(module, 'weight')
    
    for e in range(2):
        s = "Simplification #{}".format(e)
        print("#"*len(s))
        print(s)
        print("#"*len(s))
        
        model.eval()
        y_src = model(x)
        
        propagate_bias(model, zeros, pinned_out)
        model.eval()
        y_prop = model(x)
        
        print("Bias propagation")
        print("Max abs diff: ", (y_src - y_prop).abs().max().item())
        print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
        print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
        
        print("")
        
        remove_zeroed(model, zeros, pinned_out)
        model.eval()
        y_prop = model(x)
        
        print("Channel removal")
        print("Max abs diff: ", (y_src - y_prop).abs().max().item())
        print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
        print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
        
        print("")
