import random

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad
from tqdm import tqdm


def measure_matmul(x, idx, target_ch):
    # Preallocate
    eye = torch.eye(x.shape[1]).unsqueeze(0).to(device)
    idx = idx[None, :, None].expand(eye.shape)
    zeros = torch.zeros(x.shape[1], target_ch).unsqueeze(0).to(device)
    target = torch.scatter(zeros.permute(0, 2, 1), 1, idx, eye).permute(0, 2, 1)
    
    # Measure time
    starter.record()
    x_flat = x.view(x.shape[0], x.shape[1], -1)  # inference
    expanded_x = torch.matmul(target.permute(0, 2, 1), x_flat).view(x.shape[0], -1, x.shape[2], x.shape[3])  # inference
    ender.record()
    torch.cuda.synchronize()
    
    return starter.elapsed_time(ender), expanded_x


def measure_scatter(x, idx, target_ch):
    # Preallocate
    zeros = torch.zeros(1, target_ch, *x.shape[2:]).to(device)
    
    # Measure time
    starter.record()
    zeros = zeros.expand(x.shape[0], *zeros.shape[1:])
    idx = idx[None, :, None, None].expand(x.shape)
    expanded_x = torch.scatter(zeros, 1, idx, x)
    ender.record()
    torch.cuda.synchronize()
    
    return starter.elapsed_time(ender), expanded_x


def measure_select(x, idx, target_ch):
    # Preallocate
    idxs = []
    current = 0
    for i in range(target_ch):
        if i in idx:
            idxs.append(current)
            current += 1
        else:
            idxs.append(x.shape[1] - 1)
    idxs = torch.tensor(idxs, device=x.device)
    
    # Measure time
    starter.record()
    x = pad(x, (0, 0, 0, 0, 0, 1))
    expanded_x = torch.index_select(x, 1, idxs)
    ender.record()
    torch.cuda.synchronize()
    
    return starter.elapsed_time(ender), expanded_x


if __name__ == '__main__':
    # plot()
    times = []
    target_ch = 256
    simplified_ch = target_ch // 2
    device = "cuda"
    
    times_dict = {"matmul": [], "scatter": [], "select": []}
    shape = 40
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    for bs in tqdm(range(1, 257)):
        x = torch.randn(bs, simplified_ch, shape, shape).to(device)  # inference (can ignore, is output of conv)
        idx = torch.tensor(sorted(random.sample(range(target_ch), simplified_ch))).to(device)  # preallocate
        elapsed_mm, result_mm = measure_matmul(x, idx, target_ch)
        
        x = torch.randn(bs, simplified_ch, shape, shape).to(device)  # inference (can ignore, is output of conv)
        idx = torch.tensor(sorted(random.sample(range(target_ch), simplified_ch))).to(device)  # preallocate
        elapsed_scatter, result_scatter = measure_scatter(x, idx, target_ch)
        
        x = torch.randn(bs, simplified_ch, shape, shape).to(device)  # inference (can ignore, is output of conv)
        idx = torch.tensor(sorted(random.sample(range(target_ch), simplified_ch))).to(device)  # preallocate
        elapsed_select, result_select = measure_select(x, idx, target_ch)
        
        # assert (torch.equal(result_mm, result_scatter)
        #         and (torch.equal(result_mm, result_select))
        #         and (torch.equal(result_scatter, result_select)))
        
        times_dict["matmul"].append(elapsed_mm)
        times_dict["scatter"].append(elapsed_scatter)
        times_dict["select"].append(elapsed_select)
    
    for k in times_dict:
        plt.plot(times_dict[k][1:], label=k)
    
    plt.yscale("log")
    plt.xlabel("BS")
    plt.ylabel("Time")
    plt.legend()
    plt.savefig("plots.png", dpi=300)
