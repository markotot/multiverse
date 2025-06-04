import numpy as np
import torch
from einops import einops
from hydra import compose, initialize
from hydra.utils import instantiate
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv
import gymnasium as gym

from src.environments.utils import make_training_env, make_collector_env, make_evaluation_env
from src.world_models.iris_world_model.iris_env import IrisEnv
from src.world_models.iris_world_model.networks.world_model import WorldModel
from src.world_models.iris_world_model.utils import extract_state_dict


def initialize_atari_envs():
    import ale_py
    gym.register_envs(ale_py)



class AtariGymEnv:
    def __init__(self, env_id, num_envs, type='training'):

        self.num_envs = num_envs
        if type == 'training':
            self.envs = gym.vector.SyncVectorEnv([make_training_env(env_id) for _ in range(self.num_envs)])
        elif type == 'evaluation':
            self.envs = gym.vector.SyncVectorEnv([make_evaluation_env(env_id) for _ in range(self.num_envs)])
        else:
            raise ValueError('Atari Gym Env - unknown env type - pick either "training" or "evaluation"')

        self.action_space = self.envs.action_space
        self.single_action_space = self.envs.single_action_space

        self.observation_space = self.envs.observation_space
        self.single_observation_space = self.envs.single_observation_space



    def reset(self):
        return self.envs.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs.step(action)
        done = np.logical_or(terminated, truncated)
        return obs, reward, done, info


class AtariWMEnv:
    def __init__(self, env_id, single_wm_cfg, device):

        self.env_id = env_id
        x = gym.vector.SyncVectorEnv([make_collector_env(self.env_id) for _ in range(1)])
        self.env = self.initialize_world_model(single_wm_cfg, x.single_action_space.n, device)

        self.num_envs = self.env.training_cfg.world_model.batch_num_samples
        self.initialization_env = gym.vector.SyncVectorEnv([make_collector_env(self.env_id) for _ in range(self.num_envs)])

        self.single_action_space = self.initialization_env.single_action_space
        obs_shape = self.initialization_env.observation_space['greyscale'].shape
        self.single_observation_space  = np.zeros(shape=(self.env.frame_stack_size, *obs_shape[1:-1]))


    def reset(self):

        obs_stack = np.zeros(shape=(self.num_envs, self.env.frame_stack_size, 64, 64, 3), dtype=np.float32)
        observations, _ = self.initialization_env.reset()
        obs_stack[:, 0] = observations['rgb']
        # pick random numbers
        for i in range(1, self.env.frame_stack_size):
            random_actions = np.random.randint(0, self.single_action_space.n, size=self.num_envs)
            next_obs, _, _, _, _ = self.initialization_env.step(random_actions)
            obs_stack[:, i] = next_obs['rgb']

        greyscale_obs, info = self.env.reset(obs_stack)
        return greyscale_obs, info

    def step(self, actions):
        return self.env.env_step(actions)

    def initialize_world_model(self, single_wm_cfg, single_action_space, device) -> list[IrisEnv]:

        if single_wm_cfg.type == "iris":

            wm_transformer_parts = single_wm_cfg.transformer.split("/")
            wm_tokenizer_parts = single_wm_cfg.tokenizer.split("/")

            # Load the tokenizer and transformer configurations
            tokenizer_path = f"../../config/world_models/{wm_tokenizer_parts[0]}/{wm_tokenizer_parts[1]}"
            tokenizer_filename = f"{wm_tokenizer_parts[2]}.yaml"
            transformer_path = f"../../config/world_models/{wm_transformer_parts[0]}/{wm_transformer_parts[1]}"
            transformer_filename = f"{wm_transformer_parts[2]}.yaml"

            with initialize(version_base="1.3.2", config_path="../../config/world_models/iris"):
                iris_cfg = compose(config_name="iris.yaml")
            with initialize(version_base="1.3.2", config_path=tokenizer_path):
                tokenizer = instantiate(compose(config_name=tokenizer_filename))
            with initialize(version_base="1.3.2", config_path=transformer_path):
                transformer_cfg = instantiate(compose(config_name=transformer_filename))

            # Initialise the world model
            world_model = WorldModel(name=single_wm_cfg.name,
                                     obs_vocab_size=tokenizer.vocab_size,
                                     act_vocab_size=single_action_space,
                                     config=transformer_cfg,
                                     )

            iris_world_model = IrisEnv(name=single_wm_cfg.name,
                                       device=device,
                                       tokenizer=tokenizer,
                                       world_model=world_model,
                                       training_cfg=iris_cfg.training)
            if single_wm_cfg.load_checkpoint:
                env_name = self.env_id.split("NoFrameskip")[0]
                self.load_wm_checkpoint(iris_world_model, f"./checkpoint/iris/{env_name.lower()}/{env_name}.pt", device)

            iris_world_model.to(device)


        return iris_world_model

    def load_wm_checkpoint(self, wm, path_to_checkpoint, device):

        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)

        tokenizer = extract_state_dict(agent_state_dict, 'tokenizer')
        wm.tokenizer.load_state_dict(tokenizer)

        world_model = extract_state_dict(agent_state_dict, 'world_model')
        wm.world_model.load_state_dict(world_model)

    def save_checkpoint(self, path):
        self.env.save_model(path)

