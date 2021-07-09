import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tabulate import tabulate
from torchvision.models.squeezenet import SqueezeNet

import simplify
import utils
from simplify import fuse
from tests.benchmark_models import models

from torch.autograd import profiler


def profile_model(model, input, rows=10, cuda=False):
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=cuda) as prof:
        with profiler.record_function("model_inference"):
            model(input)

    return str(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=rows))

device = 'cpu'


@torch.no_grad()
def run_pruning(architecture, amount):
    print('\n----', architecture.__name__, '----')

    pretrained = True
    if architecture.__name__ in ["shufflenet_v2_x1_5",
                                 "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
        pretrained = False

    im = torch.randint(0, 256, (1, 3, 224, 224))
    x = (im / 255.).to(device)

    dense_time, pruned_time, simplified_time = [], [], []
    model = architecture(pretrained=pretrained).to(device)
    model.eval()

    for i in range(10):
        with torch.no_grad():
            start = time.perf_counter()
            y_src = model(x)
        dense_time.append(time.perf_counter() - start)

    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue

        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=amount, dim=0)
            prune.remove(module, 'weight')

    for i in range(100):
        with torch.no_grad():
            start = time.perf_counter()
            y_src = model(x)
        pruned_time.append(time.perf_counter() - start)

    profiled = profile_model(
        model, torch.randn(
            (1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'w') as f:
        f.write('-- THRESHOLDED --\n')
        f.write(profiled)

    print(
        '=> Full model inference time:',
        np.mean(pruned_time),
        np.std(pruned_time))

    model = model.to('cpu')
    bn_folding = utils.get_bn_folding(model)
    model = fuse(model, bn_folding)
    model = simplify.simplify(model, torch.zeros(
        (1, 3, 224, 224)), bn_folding=bn_folding)
    model = model.to(device)

    for i in range(100):
        with torch.no_grad():
            start = time.perf_counter()
            y_simplified = model(x)
        simplified_time.append(time.perf_counter() - start)

    profiled = profile_model(
        model, torch.randn(
            (1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'a') as f:
        f.write('\n\n -- SIMPLIFIED --\n')
        f.write(profiled)

    print(
        '=> Simplified model inference time:',
        np.mean(simplified_time),
        np.std(simplified_time))
    print('Allclose logits:', torch.allclose(y_src, y_simplified))
    print(
        'Equal predictions:', torch.equal(
            y_src.argmax(
                dim=1), y_simplified.argmax(
                dim=1)))
    print(
        f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_simplified.argmax(dim=1)).sum()}/{y_simplified.shape[0]}')

    return dense_time, pruned_time, simplified_time


if __name__ == '__main__':
    amount = 0.5

    table = []
    for architecture in models:
        try:
            d_time, p_time, s_time = run_pruning(architecture, amount)
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

    root = pathlib.Path(__file__).parent.resolve()

    index_re = re.compile(
        r"<!\-\- benchmark starts \-\->.*<!\-\- benchmark ends \-\->",
        re.DOTALL)

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
