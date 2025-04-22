import random
import torch
import numpy as np


def seed_all(seed) -> object:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return (recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def recursive_split(d, idx):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = v[idx]
        elif isinstance(v, str):
            out[k] = v[idx]
        elif isinstance(v, dict):
            out[k] = recursive_split(v, idx)
        else:
            out[k] = v
    return out