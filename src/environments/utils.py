import cv2
import numpy as np
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv
import gymnasium as gym

def initialize_atari_envs():
    import ale_py
    gym.register_envs(ale_py)


def make_training_env(env_id):
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

def make_collector_env(env_id):
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

        rgb_shape = (64, 64, 3)
        greyscale_shape = (84, 84, 1)
        env = DualObservationWrapper(env, rgb_shape, greyscale_shape)

        return env

    return thunk


class DualObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb_shape, greyscale_shape):
        super().__init__(env)
        self.rgb_shape = rgb_shape
        self.greyscale_shape = greyscale_shape
        self.observation_space = gym.spaces.Dict({
            'rgb': gym.spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8),
            'greyscale': gym.spaces.Box(low=0, high=255, shape=greyscale_shape, dtype=np.uint8)
        })

    def observation(self, obs):
        rgb = cv2.resize(obs, (self.rgb_shape[0], self.rgb_shape[1]), interpolation=cv2.INTER_AREA)
        greyscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        greyscale = cv2.resize(greyscale, (self.greyscale_shape[0], self.greyscale_shape[1]), interpolation=cv2.INTER_AREA)
        greyscale = np.expand_dims(greyscale, axis=-1)
        return {'rgb': rgb, 'greyscale': greyscale}