import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize, compose

from src.environments.training_atari_env import AtariWMEnv

if __name__ == "__main__":

    with initialize(version_base="1.3.2", config_path="./config/trainer"):
        cfg = compose(config_name="trainer.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda is True else "cpu")

    for wm in cfg.world_models:
        env = AtariWMEnv(cfg.env_id, cfg.training.num_envs, wm, device)
        o = env.reset()
        print(o)
        actions = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        o, r, d, i = env.step(actions)
        print(o, r, d, i)