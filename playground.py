import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize, compose

from src.environments.training_atari_env import AtariWMEnv

if __name__ == "__main__":

    x = [1, 2, 3, 4, 5, 6, 7, 8]
