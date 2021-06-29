import torch
from torch import nn


@torch.no_grad()
def architecture_stat(model, modules=(nn.modules.Linear, nn.modules.Conv2d, nn.modules.ConvTranspose2d, nn.modules.BatchNorm2d)):
    """
    Evaluate different statistics relative to the network's architecture.
    :param model: PyTorch model.
    :param modules: Tuple of modules to be included in the calculations.
    :return: Dictionary containing:
                layer_connection_max -> Maximum number of connections for each layer
                layer_connection_mean -> Mean number of connections for each layer
                layer_connection_min -> Minimum number of connections for each layer
                layer_connection_std -> STD of the connection for each layer
                layer_neuron_non_zero -> Number of non-zero neurons for each layer
                layer_neuron_non_zero_perc -> Percentage of non-zero neurons for each layer
                layer_neuron_ratio -> Compression ratio of the neurons for each layer
                layer_param_non_zero -> Number of non-zero parameters for each layer
                layer_param_non_zero_perc -> Percentage of non-zero parameters for each layer
                layer_param_ratio -> Compression ratio of the parameters for each layer
                network_neuron_non_zero -> Number of non-zero neurons for the whole model
                network_neuron_non_zero_perc -> Percentage of non-zero neurons for the whole model
                network_neuron_ratio -> Compression ration of the neurons for the whole model
                network_neuron_total -> Total number of neurons (zero + non-zero) for the whole model
                network_param_non_zero -> Number of non-zero parameters for the whole model
                network_param_non_zero_perc -> Percentage of non-zero parameters for the whole model
                network_param_ratio -> Compression ratio of the parameters for the whole model
                network_param_total -> Total number of parameters (zero + non-zero) for the whole model
    """
    layer_param_total = {}
    layer_param_non_zero = {}
    layer_param_non_zero_perc = {}
    layer_param_ratio = {}

    layer_neuron_total = {}
    layer_neuron_non_zero = {}
    layer_neuron_non_zero_perc = {}
    layer_neuron_ratio = {}

    layer_connection_max = {}
    layer_connection_min = {}
    layer_connection_mean = {}
    layer_connection_std = {}

    for n_m, mo in model.named_modules():
        if isinstance(mo, modules):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                layer_param_total[name] = p.numel()
                layer_param_non_zero[name] = torch.nonzero(p, as_tuple=False).shape[0]
                layer_param_non_zero_perc[name] = layer_param_non_zero[name] / layer_param_total[name] * 100
                layer_param_ratio[name] = (layer_param_total[name] / layer_param_non_zero[name]) \
                    if (layer_param_non_zero[name] != 0) else -1

                if 'weight' in name:
                    if isinstance(mo, nn.modules.Linear):
                        params_sum = torch.abs(p).sum(dim=1)
                    elif isinstance(mo, nn.modules.Conv2d):
                        params_sum = torch.abs(p).sum(dim=(1, 2, 3))
                    elif isinstance(mo, nn.modules.ConvTranspose2d):
                        params_sum = torch.abs(p).sum(dim=(0, 2, 3))
                    else:
                        params_sum = torch.abs(p)

                    layer_neuron_total[name] = params_sum.shape[0]
                    layer_neuron_non_zero[name] = torch.nonzero(params_sum, as_tuple=False).shape[0]
                    layer_neuron_non_zero_perc[name] = layer_neuron_non_zero[name] / layer_neuron_total[name] * 100
                    layer_neuron_ratio[name] = (layer_neuron_total[name] / layer_neuron_non_zero[name]) \
                        if (layer_neuron_non_zero[name] != 0) else -1

                    connections_count = torch.where(p != 0, torch.ones_like(p), torch.zeros_like(p))

                    if isinstance(mo, nn.modules.Linear):
                        connections_count = connections_count.sum(dim=1)
                    elif isinstance(mo, nn.modules.Conv2d):
                        connections_count = connections_count.sum(dim=(1, 2, 3))
                    elif isinstance(mo, nn.modules.ConvTranspose2d):
                        connections_count = connections_count.sum(dim=(0, 2, 3))
                    else:
                        connections_count = connections_count

                    connections_count = connections_count[connections_count != 0]

                    layer_connection_max[name] = connections_count.max() if connections_count.numel() > 0 else 0
                    layer_connection_min[name] = connections_count.min() if connections_count.numel() > 0 else 0
                    layer_connection_mean[name] = connections_count.mean() if connections_count.numel() > 0 else 0
                    layer_connection_std[name] = connections_count.std() if connections_count.numel() > 0 else 0

    network_param_total = sum(layer_param_total.values())
    network_param_non_zero = sum(layer_param_non_zero.values())
    network_param_non_zero_perc = network_param_non_zero / network_param_total * 100
    network_param_ratio = network_param_total / network_param_non_zero if (network_param_non_zero != 0) else -1

    network_neuron_total = sum(layer_neuron_total.values())
    network_neuron_non_zero = sum(layer_neuron_non_zero.values())
    network_neuron_non_zero_perc = network_neuron_non_zero / network_neuron_total * 100
    network_neuron_ratio = network_neuron_total / network_neuron_non_zero if (network_neuron_non_zero != 0) else -1

    return {"layer_connection_max": layer_connection_max,
            "layer_connection_mean": layer_connection_mean,
            "layer_connection_min": layer_connection_min,
            "layer_connection_std": layer_connection_std,
            "layer_neuron_non_zero": layer_neuron_non_zero,
            "layer_neuron_non_zero_perc": layer_neuron_non_zero_perc,
            "layer_neuron_ratio": layer_neuron_ratio,
            "layer_neuron_total": layer_neuron_total,
            "layer_param_non_zero": layer_param_non_zero,
            "layer_param_non_zero_perc": layer_param_non_zero_perc,
            "layer_param_ratio": layer_param_ratio,
            "layer_param_total": layer_param_total,
            "network_neuron_non_zero": network_neuron_non_zero,
            "network_neuron_non_zero_perc": network_neuron_non_zero_perc,
            "network_neuron_ratio": network_neuron_ratio,
            "network_neuron_total": network_neuron_total,
            "network_param_non_zero": network_param_non_zero,
            "network_param_non_zero_perc": network_param_non_zero_perc,
            "network_param_ratio": network_param_ratio,
            "network_param_total": network_param_total}
