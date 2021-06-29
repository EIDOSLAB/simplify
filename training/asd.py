import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.models import resnet50, resnet18

import utils
from simplify import propagate_bias, remove_zeroed, fuse
from training.stats import architecture_stat

if __name__ == '__main__':
    utils.set_seed(3)
    model = resnet18(True)

    bn_folding = utils.get_bn_folding(model)
    model = fuse(model, bn_folding)
    model.eval()
    pinned_out = utils.get_pinned_out(model)

    pruning_stat = architecture_stat(model)
    print(pruning_stat["network_param_non_zero_perc"])

    im = torch.randint(0, 256, (10, 3, 224, 224))
    x = im / 255.
    
    zeros = torch.zeros(1, *im.shape[1:])
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, name="weight", amount=amount, dim=0)
            prune.remove(module, 'weight')
    
    for e in range(2):
        
        pruning_stat = architecture_stat(model)
        print(pruning_stat["network_param_non_zero_perc"])
        
        model.eval()
        y_src = model(x)

        propagate_bias(model, zeros, pinned_out)

        model.eval()
        y_prop = model(x)

        print("Max abs diff: ", (y_src - y_prop).abs().max().item())
        print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
        print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
        
        remove_zeroed(model, zeros, pinned_out)
        
        model.eval()
        y_prop = model(x)

        print("Max abs diff: ", (y_src - y_prop).abs().max().item())
        print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
        print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
        
        # print(model)
