import numpy as np
import torch

def data_loader(x: np.array, batch_size: int, context_length: int, device: str):
    idx = torch.randint(low=0, high=len(x)-context_length, size=(batch_size,))
    
    X = [torch.from_numpy(x[i: i+context_length].astype(np.int64)) for i in idx]
    Y = [torch.from_numpy(x[i+1: i+context_length+1].astype(np.int64)) for i in idx]

    X = torch.stack(X)
    Y = torch.stack(Y)

    if "cuda" in device:
        X = X.pin_memory().to(device, non_blocking=True)
        Y = Y.pin_memory().to(device, non_blocking=True)
    else:
        X = X.to(device)
        Y = Y.to(device)

    return X, Y