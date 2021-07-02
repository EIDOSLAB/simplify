from torch.autograd import profiler


def profile_model(model, input, rows=10, cuda=False):
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=cuda) as prof:
        with profiler.record_function("model_inference"):
            model(input)

    return str(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=rows))
