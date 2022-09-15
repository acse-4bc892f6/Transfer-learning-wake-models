import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set all the random seeds to a fixed value to ensure reproducible results.

    Args:
        seed (int): Random number seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True