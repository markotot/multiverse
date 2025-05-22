import numpy as np
import torch
import dataclasses

@dataclasses.dataclass
class Dataset:
    """
    Shape is (num_envs, frames_per_env, *)
    """
    greyscale_buffer: np.ndarray
    rgb_buffer: np.ndarray
    action_buffer: np.ndarray
    reward_buffer: np.ndarray
    terminated_buffer: np.ndarray
    truncated_buffer: np.ndarray
    logprobs_buffer: np.ndarray
    value_buffer: np.ndarray

    def append(self, new_data_buffer):
        self.greyscale_buffer = np.concatenate([self.greyscale_buffer, new_data_buffer.greyscale_buffer], axis=1)
        self.rgb_buffer = np.concatenate([self.rgb_buffer, new_data_buffer.rgb_buffer], axis=1)
        self.action_buffer = np.concatenate([self.action_buffer, new_data_buffer.action_buffer], axis=1)
        self.reward_buffer = np.concatenate([self.reward_buffer, new_data_buffer.reward_buffer], axis=1)
        self.terminated_buffer = np.concatenate([self.terminated_buffer, new_data_buffer.terminated_buffer], axis=1)
        self.truncated_buffer = np.concatenate([self.truncated_buffer, new_data_buffer.truncated_buffer], axis=1)
        self.logprobs_buffer = np.concatenate([self.logprobs_buffer, new_data_buffer.logprobs_buffer], axis=1)
        self.value_buffer = np.concatenate([self.value_buffer, new_data_buffer.value_buffer], axis=1)

    def sample_batch(self, num_samples, sequence_length, device):

        max_value = self.greyscale_buffer.shape[1] - sequence_length

        episodes = np.random.choice(self.greyscale_buffer.shape[0], size=num_samples, replace=True)
        start_indices = np.random.choice(max_value, size=num_samples, replace=False)
        end_indices = start_indices + sequence_length

        return self.get_data(episodes, start_indices, end_indices, device)

    def get_data(self, episodes, start_indices, end_indices, device):

        batch = {
            'greyscale': self.get_buffer_slices(self.greyscale_buffer, episodes, start_indices, end_indices, device),
            'rgb': self.get_buffer_slices(self.rgb_buffer, episodes, start_indices, end_indices, device),
            'actions': self.get_buffer_slices(self.action_buffer, episodes, start_indices, end_indices, device),
            'rewards': self.get_buffer_slices(self.reward_buffer, episodes, start_indices, end_indices, device),
            'terminateds': self.get_buffer_slices(self.terminated_buffer, episodes, start_indices, end_indices, device),
            'truncateds': self.get_buffer_slices(self.truncated_buffer, episodes, start_indices, end_indices, device),
            'logprobs': self.get_buffer_slices(self.logprobs_buffer, episodes, start_indices, end_indices, device),
            'values': self.get_buffer_slices(self.value_buffer, episodes, start_indices, end_indices, device)

        }
        return batch

    @staticmethod
    def get_buffer_slices(buffer, episodes, start_indices, end_indices, device):
        # Final result will store the slices
        result = []

        for i in range(len(episodes)):
            # Get the episode index
            ep = episodes[i]
            # Get start and end indices for this episode
            start = start_indices[i]
            end = end_indices[i]

            # Extract the slice for this episode
            # This gets the specific episode and the range from start to end
            slice_data = buffer[ep, start:end]
            result.append(slice_data)

        np_result = np.array(result)
        result = torch.from_numpy(np_result).to(device)
        return result