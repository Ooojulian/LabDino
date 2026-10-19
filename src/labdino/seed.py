import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fija las semillas para que el entrenamiento en CPU sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
