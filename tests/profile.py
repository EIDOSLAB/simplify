import torch
from torch.autograd import profiler
from torchvision.models import resnet18


def profile_model(model, input, rows=10):
    with profiler.profile(profile_memory=True) as prof:
        with profiler.record_function("model_inference"):
            model(input)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=rows))