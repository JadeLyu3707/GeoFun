# @author: yl12783, Nov. 2026, New York

import os
import random
import numpy as np


def set_seed(seed=0xD3):
    random.seed(seed)
    np.random.seed(seed)


def set_out(cfg):
    experiment_folder = os.path.join(cfg.root_path, "experiments", cfg.experiment)
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder
