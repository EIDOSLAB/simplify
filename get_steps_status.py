import torch
from tabulate import tabulate

import fuser
import utils
from simplify import __propagate_bias, __remove_zeroed
from tests.benchmark_models import models

if __name__ == '__main__':
    utils.set_seed(3)
    
    x = torch.randint(0, 256, (256, 3, 224, 224))
    x = x.float() / 255.
    
    table = []
    for architecture in models:
        print(architecture.__name__)
        model = architecture(True, progress=False)
        model.eval()
        y_src = model(x)
        
        print("BatchNorm Folding")
        try:
            model = fuser.fuse(model)
            exception = None
        except Exception as e:
            exception = e.__class__
        y_dest = model(x)
        passed = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
        table.append([architecture.__name__, "BatchNorm Folding", ":heavy_check_mark:" if passed else ":x:",
                      "" if exception is None else exception])
        if not passed:
            continue
        
        pinned_out = utils.get_pinned_out(model)
        
        print("Bias Propagation")
        try:
            __propagate_bias(model, x, pinned_out)
            exception = None
        except Exception as e:
            exception = e.__class__
        y_dest = model(x)
        passed = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
        table.append([architecture.__name__, "Bias Propagation", ":heavy_check_mark:" if passed else ":x:",
                      "" if exception is None else exception])
        if not passed:
            continue
        
        print("Simplification")
        try:
            __remove_zeroed(model, pinned_out)
            exception = None
        except Exception as e:
            exception = e.__class__
        y_dest = model(x)
        passed = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
        table.append([architecture.__name__, "Simplification", ":heavy_check_mark:" if passed else ":x:",
                      "" if exception is None else exception])
        if not passed:
            continue
    
    table = tabulate(table, headers=['Architecture', 'Step', 'Pass', 'Exception'], tablefmt='github')
    print(table)
