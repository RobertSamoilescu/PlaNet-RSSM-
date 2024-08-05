import torch
import random
import numpy as np


def set_seed(seed: int = 13) -> None:
    """
    Sets a seed to ensure reproducibility
    Parameters
    ----------
    seed
        seed to be set
    """

    # torch related
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # others
    np.random.seed(seed)
    random.seed(seed)
