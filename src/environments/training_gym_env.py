import numpy as np
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv
import gymnasium as gym

def initialize_atari_envs():
    import ale_py
    gym.register_envs(ale_py)

def make_env(env_id):
    def thunk():

        if env_id not in gym.envs.registry:
            print("No env name found, trying to add to registry")
            initialize_atari_envs()

        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


class AtariGymEnv:
    def __init__(self, env_id, num_envs):

        self.envs = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])

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


if __name__ == "__main__":
    env = AtariGymEnv("PongNoFrameskip-v4", 4)
    o = env.reset()
    print(o)
    actions = np.array([0, 0, 0, 0])
    o, r, d, i = env.step(actions)
    print(o, r, d, i)