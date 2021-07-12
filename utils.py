import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, kwargs)
        return time.perf_counter() - start
    return wrapper