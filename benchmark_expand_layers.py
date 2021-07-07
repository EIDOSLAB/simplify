import random
import time

import torch
import wandb
from torch import nn
from torch.nn.functional import pad
from tqdm import tqdm

from utils import set_seed


def measure_base(conv_module, x):
    if device == torch.device("cuda"):
        starter.record()
    else:
        start = time.perf_counter()
        
    x = conv_module(x)
    
    if device == torch.device("cuda"):
        ender.record()
        torch.cuda.synchronize()
    else:
        end = time.perf_counter()
    
    return starter.elapsed_time(ender) if device == torch.device("cuda") else end - start


def measure_matmul(conv_module, x, idx, target_ch, conv_bias):
    # Preallocate
    eye = torch.eye(conv_module.weight.shape[0]).unsqueeze(0).to(device)
    idx = idx[None, :, None].expand(eye.shape)
    zeros = torch.zeros(conv_module.weight.shape[0], target_ch).unsqueeze(0).to(device)
    target = torch.scatter(zeros.permute(0, 2, 1), 1, idx, eye).permute(0, 2, 1)
    
    # Measure time
    if device == torch.device("cuda"):
        starter.record()
    else:
        start = time.perf_counter()
    
    x = conv_module(x)
    x_flat = x.view(x.shape[0], x.shape[1], -1)  # inference
    expanded_x = torch.matmul(target.permute(0, 2, 1), x_flat).view(x.shape[0], -1, x.shape[2], x.shape[3])  # inference
    expanded_x += conv_bias
    
    if device == torch.device("cuda"):
        ender.record()
        torch.cuda.synchronize()
    else:
        end = time.perf_counter()
    
    return starter.elapsed_time(ender) if device == torch.device("cuda") else end - start


def measure_scatter(conv_module, x, simplified_x, idx, target_ch, conv_bias):
    # Preallocate
    zeros = torch.zeros(1, target_ch, *simplified_x.shape[2:]).to(device)
    
    # Measure time
    if device == torch.device("cuda"):
        starter.record()
    else:
        start = time.perf_counter()
    
    x = conv_module(x)
    zeros = zeros.expand(x.shape[0], *zeros.shape[1:])
    idx = idx[None, :, None, None].expand(x.shape)
    expanded_x = torch.scatter(zeros, 1, idx, x)
    expanded_x += conv_bias
    
    if device == torch.device("cuda"):
        ender.record()
        torch.cuda.synchronize()
    else:
        end = time.perf_counter()
    
    return starter.elapsed_time(ender) if device == torch.device("cuda") else end - start


def measure_select(conv_module, x, idx, target_ch, conv_bias):
    # Preallocate
    idxs = []
    current = 0
    for i in range(target_ch):
        if i in idx:
            idxs.append(current)
            current += 1
        else:
            idxs.append(conv_module.weight.shape[0])
    idxs = torch.tensor(idxs, device=x.device)
    
    # Measure time
    if device == torch.device("cuda"):
        starter.record()
    else:
        start = time.perf_counter()
    
    x = conv_module(x)
    x = pad(x, (0, 0, 0, 0, 0, 1))
    expanded_x = torch.index_select(x, 1, idxs)
    expanded_x += conv_bias
    
    if device == torch.device("cuda"):
        ender.record()
        torch.cuda.synchronize()
    else:
        end = time.perf_counter()
    
    return starter.elapsed_time(ender) if device == torch.device("cuda") else end - start


def measure_indexing(conv_module, x, idx, target_ch, conv_bias):
    # Preallocate
    idxs = []
    current = 0
    for i in range(target_ch):
        if i in idx:
            idxs.append(current)
            current += 1
        else:
            idxs.append(conv_module.weight.shape[0])
    idxs = torch.tensor(idxs, device=x.device)
    
    # Measure time
    if device == torch.device("cuda"):
        starter.record()
    else:
        start = time.perf_counter()
    
    x = conv_module(x)
    x = pad(x, (0, 0, 0, 0, 0, 1))
    expanded_x = x[:, idxs]
    expanded_x += conv_bias
    
    if device == torch.device("cuda"):
        ender.record()
        torch.cuda.synchronize()
    else:
        end = time.perf_counter()
    
    return starter.elapsed_time(ender) if device == torch.device("cuda") else end - start


if __name__ == '__main__':
    set_seed(0)
    total_ch = 256
    device = "cuda"
    shape = 40
    tot_batches = 257
    
    for pruning in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        
        config = {
            "device":      device,
            "map_shape":   shape,
            "total_ch":    total_ch,
            "pruning":     pruning,
            "tot_batches": tot_batches,
        }
        wandb.init(entity='eidos', project="Simplify", config=config)
        
        simplified_ch = int(total_ch * pruning)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        base_time = []
        mm_time = []
        scatter_time = []
        select_time = []
        indexing_time = []
        
        for bs in tqdm(range(1, tot_batches)):
            x = torch.randn(bs, total_ch, shape, shape).to(device)  # inference (can ignore, is output of conv)
            to_prune_idx = torch.tensor(sorted(random.sample(range(total_ch), total_ch - simplified_ch))).to(
                device)  # preallocate
            
            base_conv = nn.Conv2d(total_ch, total_ch, (3, 3))
            base_conv = base_conv.to(device)
            
            with torch.no_grad():
                base_conv.weight[to_prune_idx] *= 0
                base_conv.bias[to_prune_idx] *= 0
            
            for _ in range(3):
                pruned_output = base_conv(x)
            
            ########
            # BASE #
            ########
            
            base_time.append(measure_base(base_conv, x))
            
            nonzero_idx = ~(base_conv.weight.view(base_conv.weight.shape[0], -1).sum(dim=1) == 0)
            conv_bias = base_conv.bias.data.clone()[None, :, None, None].expand_as(pruned_output)
            
            with torch.no_grad():
                base_conv.weight = torch.nn.Parameter(base_conv.weight[nonzero_idx])
                setattr(base_conv, "bias", None)
            
            base_conv = base_conv.to(device)
            
            for _ in range(3):
                base_output = base_conv(x)
            
            idx = torch.where(nonzero_idx)[0]
            
            ######
            # MM #
            ######
            
            mm_time.append(measure_matmul(base_conv, x, idx, total_ch, conv_bias))
            
            ###########
            # SCATTER #
            ###########
            
            scatter_time.append(measure_scatter(base_conv, x, base_output, idx, total_ch, conv_bias))
            
            ##########
            # SELECT #
            ##########
            
            select_time.append(measure_select(base_conv, x, idx, total_ch, conv_bias))
            
            ############
            # INDEXING #
            ############
            
            indexing_time.append(measure_indexing(base_conv, x, idx, total_ch, conv_bias))
            
            wandb.log({
                'base':     base_time[bs - 1],
                'mm':       mm_time[bs - 1],
                'scatter':  scatter_time[bs - 1],
                'select':   select_time[bs - 1],
                'indexing': indexing_time[bs - 1],
            })
        
        wandb.log({
            "Inference": wandb.plot.line_series(
                xs=[b for b in range(1, tot_batches)],
                ys=[base_time, mm_time, scatter_time, select_time, indexing_time],
                keys=["base", "mm", "scatter", "select", "indexing"],
                title="Inference Time",
                xname="Batch Size"
            )
        })
        
        wandb.finish()
