import random
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.functional import pad
from tqdm import tqdm

from utils import set_seed


def measure_matmul(x, idx, target_ch, conv_bias):
    # Preallocate
    eye = torch.eye(x.shape[1]).unsqueeze(0).to(device)
    idx = idx[None, :, None].expand(eye.shape)
    zeros = torch.zeros(x.shape[1], target_ch).unsqueeze(0).to(device)
    target = torch.scatter(zeros.permute(0, 2, 1), 1, idx, eye).permute(0, 2, 1)
    
    # Measure time
    x_flat = x.view(x.shape[0], x.shape[1], -1)  # inference
    expanded_x = torch.matmul(target.permute(0, 2, 1), x_flat).view(x.shape[0], -1, x.shape[2], x.shape[3])  # inference
    expanded_x += conv_bias
    
    return expanded_x


def measure_scatter(x, idx, target_ch, conv_bias):
    # Preallocate
    zeros = torch.zeros(1, target_ch, *x.shape[2:]).to(device)
    
    # Measure time
    zeros = zeros.expand(x.shape[0], *zeros.shape[1:])
    idx = idx[None, :, None, None].expand(x.shape)
    expanded_x = torch.scatter(zeros, 1, idx, x)
    expanded_x += conv_bias
    
    return expanded_x


def measure_select(x, idx, target_ch, conv_bias):
    # Preallocate
    idxs = []
    current = 0
    for i in range(target_ch):
        if i in idx:
            idxs.append(current)
            current += 1
        else:
            idxs.append(x.shape[1])
    idxs = torch.tensor(idxs, device=x.device)
    
    # Measure time
    x = pad(x, (0, 0, 0, 0, 0, 1))
    expanded_x = torch.index_select(x, 1, idxs)
    expanded_x += conv_bias
    
    return expanded_x


def measure_indexing(x, idx, target_ch, conv_bias):
    # Preallocate
    idxs = []
    current = 0
    for i in range(target_ch):
        if i in idx:
            idxs.append(current)
            current += 1
        else:
            idxs.append(x.shape[1])
    idxs = torch.tensor(idxs, device=x.device)
    
    # Measure time
    x = pad(x, (0, 0, 0, 0, 0, 1))
    expanded_x = x[:, idxs]
    expanded_x += conv_bias
    
    return expanded_x


if __name__ == '__main__':
    set_seed(0)
    times = []
    target_ch = 256
    simplified_ch = int(target_ch * 0.3)
    device = "cuda"
    
    times_dict = {"conv_base":            [],
                  "conv_expand_mm":       [],
                  "conv_expand_scatter":  [],
                  "conv_expand_select":   [],
                  "conv_expand_indexing": []}
    shape = 40
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    for bs in tqdm(range(1, 31)):
        x = torch.randn(bs, target_ch, shape, shape).to(device)  # inference (can ignore, is output of conv)
        to_prune_idx = torch.tensor(sorted(random.sample(range(target_ch), target_ch - simplified_ch))).to(
            device)  # preallocate
        
        base_conv = nn.Conv2d(target_ch, target_ch, (3, 3))
        base_conv = base_conv.to(device)
        
        with torch.no_grad():
            base_conv.weight[to_prune_idx] *= 0
            base_conv.bias[to_prune_idx] *= 0
        
        for _ in range(3):
            pruned_output = base_conv(x)
        
        ########
        # BASE #
        ########
        
        if device == torch.device("cuda"):
            starter.record()
        else:
            start = time.perf_counter()
        pruned_output = base_conv(x)
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        times_dict["conv_base"].append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
        
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
        
        if device == torch.device("cuda"):
            starter.record()
        else:
            start = time.perf_counter()
        simplified_output = base_conv(x)
        simplified_output_mm = measure_matmul(simplified_output, idx, target_ch, conv_bias)
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        times_dict["conv_expand_mm"].append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
        
        ###########
        # SCATTER #
        ###########
        
        if device == torch.device("cuda"):
            starter.record()
        else:
            start = time.perf_counter()
        simplified_output = base_conv(x)
        simplified_output_scatter = measure_scatter(simplified_output, idx, target_ch, conv_bias)
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        times_dict["conv_expand_scatter"].append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
        
        ##########
        # SELECT #
        ##########
        
        if device == torch.device("cuda"):
            starter.record()
        else:
            start = time.perf_counter()
        simplified_output = base_conv(x)
        simplified_output_select = measure_select(simplified_output, idx, target_ch, conv_bias)
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        times_dict["conv_expand_select"].append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
        
        ############
        # INDEXING #
        ############
        
        if device == torch.device("cuda"):
            starter.record()
        else:
            start = time.perf_counter()
        simplified_output = base_conv(x)
        simplified_output_indexing = measure_indexing(simplified_output, idx, target_ch, conv_bias)
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        times_dict["conv_expand_indexing"].append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
    
    for k in times_dict:
        plt.plot(times_dict[k][1:], label=k)
    
    plt.yscale("log")
    plt.xlabel("Batch Size")
    plt.ylabel("Forward Time (ms)")
    plt.title(f"Pruning {simplified_ch / target_ch * 100:.2f}% on {device}")
    plt.legend()
    plt.savefig(f"Benchmark_expand_{device}", dpi=300)
