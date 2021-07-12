import random
import os
import numpy as np
import torch
import time

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, kwargs)
        return time.perf_counter() - start
    return wrapper