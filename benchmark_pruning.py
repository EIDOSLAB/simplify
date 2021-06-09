import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tabulate import tabulate
from torchvision.models.squeezenet import SqueezeNet

import profile
import simplify
import utils
from tests.benchmark_models import models

device = 'cpu'


@torch.no_grad()
def run_pruning(architecture, amount):
    print('\n----', architecture.__name__, '----')
    
    pretrained = True
    if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
        pretrained = False
    
    full_time, simplified_time = [], []
    model = architecture(pretrained=pretrained).to(device)
    model.eval()
    
    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue
        
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=amount, dim=0)
            prune.remove(module, 'weight')
        
        # if isinstance(module, nn.BatchNorm2d):
        #    prune.random_unstructured(module, 'weight', amount=amount)
        #    prune.remove(module, 'weight')
    
    im = torch.randint(0, 256, (100, 3, 224, 224))
    x = (im / 255.).to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_src = model(x)
        full_time.append(time.perf_counter() - start)
    
    profiled = profile.profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'w') as f:
        f.write('-- THRESHOLDED --\n')
        f.write(profiled)
    
    print('=> Full model inference time:', np.mean(full_time), np.std(full_time))
    
    pinned_out = utils.get_pinned_out(model)
    
    model = model.to('cpu')
    bn_folding = utils.get_bn_folding(model)
    model = simplify.simplify(model, torch.zeros((1, 3, 224, 224)), pinned_out=pinned_out, bn_folding=bn_folding)
    model = model.to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_simplified = model(x)
        simplified_time.append(time.perf_counter() - start)
    
    profiled = profile.profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'a') as f:
        f.write('\n\n -- SIMPLIFIED --\n')
        f.write(profiled)
    
    print('=> Simplified model inference time:', np.mean(simplified_time), np.std(simplified_time))
    print('Allclose logits:', torch.allclose(y_src, y_simplified))
    print('Equal predictions:', torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))
    print(
        f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_simplified.argmax(dim=1)).sum()}/{y_simplified.shape[0]}')
    
    return full_time, simplified_time


if __name__ == '__main__':
    utils.set_seed(3)
    
    amount = 0.5
    
    table = []
    for architecture in models:
        try:
            full_time, s_time = run_pruning(architecture, amount)
        except Exception as e:
            full_time, s_time = [0.], [0.]
        
        table.append([architecture.__name__, f'{np.mean(full_time):.4f}s ± {np.std(full_time):.4f}',
                      f'{np.mean(s_time):.4f}s ± {np.std(s_time):.4f}'])
    table = tabulate(table, headers=['Architecture', 'Pruned time', 'Simplified time'], tablefmt='github')
    print(table)
    
    import pathlib
    import re
    
    root = pathlib.Path(__file__).parent.resolve()
    
    index_re = re.compile(r"<!\-\- benchmark starts \-\->.*<!\-\- benchmark ends \-\->", re.DOTALL)
    
    updated = "Update timestamp " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n"
    pruning_perc = "Random structured pruning amount = " + str(amount * 100) + "%\n"
    
    index = ["<!-- benchmark starts -->", updated, pruning_perc, table, "<!-- benchmark ends -->"]
    readme = root / "README.md"
    index_txt = "\n".join(index).strip()
    readme_contents = readme.open().read()
    readme.open("w").write(index_re.sub(index_txt, readme_contents))
