import sys

import cv2
import numpy as np
import torch
from collections import defaultdict
from typing import Optional, Any, Dict, Tuple, List

import einops
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from tqdm import tqdm
from PIL import Image
from src.environments.dataset import Dataset
from src.visualization.plotting import plot_images
from src.world_models.iris_world_model.utils import configure_optimizer, rgb_to_grayscale, rescale_image


class IrisEnv(nn.Module):
    def __init__(self, name, tokenizer: torch.nn.Module, world_model: torch.nn.Module, training_cfg: DictConfig, device: torch.device):

        super().__init__()
        self.name = name

        self.device = device

        self.world_model = world_model
        self.tokenizer = tokenizer

        self.training_cfg = training_cfg

        self.optimizer_tokenizer = torch.optim.Adam(self.tokenizer.parameters(), lr=self.training_cfg.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.world_model, self.training_cfg.learning_rate,
                                                         self.training_cfg.world_model.weight_decay)

        self.current_step = 0
        self.grey_scale_size = (84, 84)
        self.frame_stack_size = 4

        # self.observation_space = (self.frame_stack_size, *self.grey_scale_size)
        # self.action_space = 4

        self.framestack_observation_tokens = np.zeros((self.frame_stack_size, self.world_model.config.tokens_per_block - 1))
        self.keys_values_wm,  self.obs_tokens, self._num_observations_tokens = None, None, None


    def train_model(self, dataset):

        cfg_tokenizer = self.training_cfg.tokenizer
        cfg_world_model = self.training_cfg.world_model

        metrics_tokenizer = self.train_component(self.tokenizer,
                                                 self.optimizer_tokenizer,
                                                 sequence_length=1,
                                                 sample_from_start=True,
                                                 dataset=dataset,
                                                 **cfg_tokenizer
                                                 )

        metrics_world_model = self.train_component(self.world_model,
                                                   self.optimizer_world_model,
                                                   sequence_length=self.world_model.config.max_blocks,
                                                   sample_from_start=True,
                                                   dataset=dataset,
                                                   tokenizer=self.tokenizer,
                                                   **cfg_world_model
                                                   )

        return [{**metrics_tokenizer, **metrics_world_model}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                        dataset: Any, batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float],
                        sequence_length: int, sample_from_start: bool, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {str(component)}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):



                # find a way to select a batch of episodes
                batched_dataset = dataset
                batch = batched_dataset.sample_batch(batch_num_samples, sequence_length, self.device)
                losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics


    def evaluate_encoder(self, dataset: Dataset):

        batch = dataset.sample_batch(4, 1, device=self.device) # sample a batch of data
        observations = batch['rgb'][:, 0, 0, :, :, :] # extract the first image of the framestack

        output_obs, obs_tokens = self.vq_encoder_only(observations)
        for i, (in_obs, out_obs) in enumerate(zip(observations, output_obs)):
            in_obs = in_obs.detach().cpu().numpy()
            in_obs = in_obs / 255.0
            out_obs = out_obs.detach().cpu().numpy().transpose(1, 2, 0)
            plot_images([in_obs, out_obs], title=f"Encoder eval:{i} -- {self.name} ", transpose=False)



    @torch.no_grad()
    def vq_encoder_only(self, observations) -> [np.ndarray, torch.Tensor]:

        observations = observations / 255.0 * 2 - 1
        observations = einops.rearrange(observations, 'b h w c -> b c h w')
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=False).tokens
        embedded_tokens = self.tokenizer.embedding(obs_tokens)  # (B, K, E)

        # (B, K, E) -> (B, E, H, W)
        h = int(np.sqrt(self.world_model.config.tokens_per_block - 1))
        z = einops.rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=h)

        decoded_obs = self.tokenizer.decode(z, should_postprocess=True)  # (B, C, H, W)
        decoded_obs = torch.clamp(decoded_obs, 0, 1)
        return decoded_obs, obs_tokens


    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    def reset_from_initial_observation(self, observation: np.ndarray) -> list:

        obs = einops.rearrange(observation, 'b fs h w c -> (b fs) c h w')
        obs = obs / 255.0 * 2 - 1  # Normalize to [-1, 1]
        obs = torch.from_numpy(obs).float().to(self.device)
        obs_tokens = self.tokenizer.encode(obs, should_preprocess=False).tokens  # (BL, K)

        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        fs = observation.shape[1]
        unstacked_obs_tokens = obs_tokens[(fs-1)::fs] # Take the most resent observation from each framestack
        _ = self.refresh_keys_values_with_initial_obs_tokens(unstacked_obs_tokens)

        self.framestack_observation_tokens = einops.rearrange(obs_tokens, '(b fs) n -> b fs n', fs=fs)
        decoded_observations = self.decode_obs_tokens(obs_tokens)
        decoded_observations = einops.rearrange(decoded_observations, '(b fs) c h w -> b fs c h w', fs=fs)

        return decoded_observations

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def decode_obs_tokens(self, obs_tokens) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(obs_tokens)     # (B, K, E)
        z = einops.rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    def reset(self, initial_observations) -> [np.ndarray, dict]:

        # Get the first observation from the buffer
        obs = self.reset_from_initial_observation(initial_observations) # Image passed through the encoder
        grayscale_obs = self.wm_obs_to_grayscale(obs)
        info = {"step": self.current_step}

        return grayscale_obs, info

    def wm_obs_to_grayscale(self, obs):

        # convert to greyscale: # (B, FS, ...)
        fs = obs.shape[1]

        luminosity_weights = torch.tensor([0.2125, 0.7154, 0.0721],  device=self.device)
        #luminosity_weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
        luminosity_weights = luminosity_weights.view(1, 3, 1, 1)  # reshape for broadcasting


        obs = einops.rearrange(obs, 'b fs c h w -> (b fs) c h w')
        grayscale_obs = torch.sum(obs * luminosity_weights, dim=1)

        # rescale to (84, 84) # (B, FS, ...)
        grayscale_obs = grayscale_obs.unsqueeze(1)
        grayscale_obs = F.interpolate(grayscale_obs, size=self.grey_scale_size, mode='bilinear', align_corners=False)
        #grayscale_obs = F.interpolate(grayscale_obs, size=self.grey_scale_size, mode='nearest')

        grayscale_obs = grayscale_obs.squeeze(1)

        grayscale_obs = einops.rearrange(grayscale_obs, '(b fs) h w -> b fs h w', fs=fs)
        return grayscale_obs

    def env_step(self, actions: np.ndarray):

        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens
        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.framestack_observation_tokens[:, -1])

        token = actions.clone().detach() if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.

            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)

        self.framestack_observation_tokens = torch.roll(self.framestack_observation_tokens, shifts=-1, dims=1)
        new_obs_tokens = torch.cat(obs_tokens, dim=1)
        self.framestack_observation_tokens[:, -1] = new_obs_tokens

        x = einops.rearrange(self.framestack_observation_tokens, 'b fs n -> (b fs) n')
        x = self.decode_obs_tokens(x)
        decoded_observations = einops.rearrange(x, '(b fs) c h w -> b fs c h w',
                                                fs=4)

        greyscale_obs = self.wm_obs_to_grayscale(decoded_observations)
        return greyscale_obs, reward, done, None

    def save_model(self, path: str):
        torch.save({
            'tokenizer': self.tokenizer.state_dict(),
            'world_model': self.world_model.state_dict(),
            'optimizer_tokenizer': self.optimizer_tokenizer.state_dict(),
            'optimizer_world_model': self.optimizer_world_model.state_dict(),
            'current_step': self.current_step
        }, path)