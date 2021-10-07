import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from torchvision.models import inception, inception_v3, SqueezeNet
from tqdm import tqdm

import os
import simplify
from tests.benchmark_models import models

device = os.environ.get('DEVICE', 'cuda')


def measure_memory(model, x, y):
    mem_total = torch.cuda.get_device_properties(0)
    mem_reserved = [torch.cuda.memory_reserved(0)]
    mem_allocated = [torch.cuda.memory_allocated(0)]

    model = model.to(device)
    with torch.enable_grad():
        output = model(x)
        loss = F.cross_entropy(output, y)
    loss.backward()
    mem_reserved.append(torch.cuda.memory_reserved(0))
    mem_allocated.append(torch.cuda.memory_allocated(0))

    del model
    del x
    del y
    torch.cuda.empty_cache()

    return mem_total, mem_reserved, mem_allocated

def main(network):
    print('=> Benchmarking', network.__name__)

    batch_size = 128
    h, w = 224, 224
    if network.__name__ == "inception_v3":
        h, w = 299, 299
        
    fake_input = torch.randint(0, 256, (batch_size, 3, h, w))
    fake_input = fake_input.float() / 255.
    fake_target = torch.randint(0, 1000, (batch_size,)).long()
    criterion = CrossEntropyLoss()
    
    prune_step = 0.05
    iterations = int(1. / prune_step)
    amount = 0.
    
    wandb.init(
        config={'arch': network.__name__, 'device': os.environ.get('DEVICE', 'cuda')},
        group="benchmark_memory"
    )
    
    for _ in tqdm(range(iterations), desc="Benchmark"):
        if amount > 1.:
            break
        
        if network.__name__ in ["inception_v3", "googlenet"]:
            model = network(False, aux_logits=False)
        else:
            model = network(True)

        if amount > 0:
            for name, module in model.named_modules():
                if isinstance(model, SqueezeNet) and 'classifier.1' in name:
                    continue
                if isinstance(module, nn.Conv2d):
                    prune.random_structured(module, 'weight', amount=amount, dim=0)
                    prune.remove(module, 'weight')

            model.eval()
            simplify.simplify(model, torch.zeros(1, 3, h, w).to(device), 
                              fuse_bn=False, training=True)
            model.train()
        

        mem_total, mem_reserved, mem_allocated = measure_memory(model)

        wandb.log({
            'amount': amount,
            'mem_reserved.0': mem_reserved[0],
            'mem_reserved.1': mem_reserved[1],
            'mem_allocated.0': mem_allocated[0],
            'mem_allocated.1': mem_allocated[1]
        })
       
        amount += prune_step
        del model
        torch.cuda.empty_cache()
    
    wandb.finish()


if __name__ == '__main__':
    for arch in models:
        main(arch)
