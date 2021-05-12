from collections import defaultdict

import torch
from EIDOSearch.pruning.simplification.fuser import fuse
from torchvision.models import resnet18


def dfs(grad_fn, modules, ops):
    if grad_fn[0] is None:
        return
    
    visited = False
    
    if hasattr(grad_fn[0], "variable"):
        if hasattr(grad_fn[0].variable, "module_name"):
            if grad_fn[0].variable.module_name in modules:
                visited = True
            modules[grad_fn[0].variable.module_name] += 1
            print(grad_fn[0].variable.module_name, visited)
        else:
            return
    
    if visited:
        return
    
    for fn in grad_fn[0].next_functions:
        if fn[0] is not None:
            if fn[0] in ops:
                ops[fn[0]].append(grad_fn[0])
            else:
                ops[fn[0]] = [grad_fn[0]]
        dfs(fn, modules, ops)


if __name__ == '__main__':
    model = resnet18()
    print(model)
    model = fuse(model)
    
    print(model)
    
    for i, (n, m) in enumerate(model.named_modules()):
        for np, p in m.named_parameters():
            if "weight" in np:
                setattr(p, "module_name", n)
    
    out = model(torch.randn(1, 3, 224, 224))
    
    modules_dict = defaultdict(int)
    ops = {}
    
    dfs((out.grad_fn,), modules_dict, ops)
    
    for k in ops:
        ops[k] = set(ops.pop(k))
    print()
