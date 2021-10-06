import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tabulate import tabulate
from torch.autograd import profiler
from torchvision.models.squeezenet import SqueezeNet

import simplify
from tests.benchmark_models import models


def profile_model(model, input, rows=10, cuda=False):
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=cuda) as prof:
        with profiler.record_function("model_inference"):
            model(input)
    
    return str(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=rows))


device = 'cpu'


@torch.no_grad()
def run_pruning(architecture, amount, mode):
    print('\n----', architecture.__name__, '----')
    
    im = torch.randint(0, 256, (1, 3, 224, 224))
    x = (im / 255.).to(device)
    
    dense_time, pruned_time, simplified_time = [], [], []
    print(architecture.__name__)
    if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
        pretrained = False
    else:
        pretrained = True
    if architecture.__name__ in ["inception_v3", "googlenet"]:
        model = architecture(pretrained, transform_input=False, aux_logits=False)
    else:
        model = architecture(pretrained)
    
    model.to(device)
    model.train(mode == "train")
    
    for i in range(101):
        with torch.no_grad():
            start = time.perf_counter()
            y_src = model(x)
            end = time.perf_counter()
        if i > 0:
            dense_time.append(end - start)
    
    print('=> Dense model inference time:',
          np.mean(dense_time),
          np.std(dense_time))
    
    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue
        
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=amount, dim=0)
            prune.remove(module, 'weight')
    
    model.train(mode == "train")
    
    for i in range(101):
        with torch.no_grad():
            start = time.perf_counter()
            y_src = model(x)
            end = time.perf_counter()
        if i > 0:
            pruned_time.append(end - start)
    
    try:
        profiled = profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
        with open(f'profile/{architecture.__name__}_{mode}.txt', 'w') as f:
            f.write('-- PRUNED --\n')
            f.write(profiled)
    except:
        pass
    
    print('=> Pruned model inference time:',
          np.mean(pruned_time),
          np.std(pruned_time))
    
    model.eval()
    model = model.to('cpu')
    model = simplify.simplify(model, torch.zeros((1, 3, 224, 224)), training=mode == "train", fuse_bn=False)
    model = model.to(device)
    
    model.train(mode == "train")
    
    for i in range(101):
        with torch.no_grad():
            start = time.perf_counter()
            y_simplified = model(x)
            end = time.perf_counter()
        if i > 0:
            simplified_time.append(end - start)
    
    try:
        profiled = profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
        with open(f'profile/{architecture.__name__}_{mode}.txt', 'w') as f:
            f.write('-- SIMPLIFIED --\n')
            f.write(profiled)
    except:
        pass
    
    print('=> Simplified model inference time:',
          np.mean(simplified_time),
          np.std(simplified_time))
    
    print('Allclose logits:', torch.allclose(y_src, y_simplified))
    print('Equal predictions:', torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))
    print(f'Correct predictions: '
          f'{torch.eq(y_src.argmax(dim=1), y_simplified.argmax(dim=1)).sum()}/{y_simplified.shape[0]}')
    
    return dense_time, pruned_time, simplified_time


if __name__ == '__main__':
    amount = 0.5
    
    for mode in ["train", "eval"]:
        print(f"\nMODE: {mode}\n")
        table = []
        for architecture in models:
            try:
                d_time, p_time, s_time = run_pruning(architecture, amount, mode)
            except Exception as e:
                raise e
                d_time, p_time, s_time = [0.], [0.], [0.]
            
            table.append([architecture.__name__,
                          f'{np.mean(d_time):.4f}s ± {np.std(d_time):.4f}',
                          f'{np.mean(p_time):.4f}s ± {np.std(p_time):.4f}',
                          f'{np.mean(s_time):.4f}s ± {np.std(s_time):.4f}'])
        table = tabulate(
            table,
            headers=[
                'Architecture',
                'Dense time',
                'Pruned time',
                'Simplified time'],
            tablefmt='github')
        print(table)
        
        import pathlib
        import re
        
        root = pathlib.Path(__file__).parent.parent.resolve()
        
        index_re = re.compile(rf"<!\-\- benchmark {mode} starts \-\->.*<!\-\- benchmark {mode} ends \-\->", re.DOTALL)
        
        updated = "Update timestamp " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n"
        pruning_perc = "Random structured pruning amount = " + \
                       str(amount * 100) + "%\n"
        
        index = [
            "<!-- benchmark starts -->",
            updated,
            pruning_perc,
            table,
            "<!-- benchmark ends -->"]
        readme = root / "README.md"
        index_txt = "\n".join(index).strip()
        readme_contents = readme.open().read()
        readme.open("w").write(index_re.sub(index_txt, readme_contents))
