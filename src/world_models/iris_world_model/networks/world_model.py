from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.iris_world_model.networks.kv_caching import KeysValues
from src.world_models.iris_world_model.networks.slicer import Embedder, Head
from src.world_models.iris_world_model.networks.tokenizer.tokenizer import Tokenizer
from src.world_models.iris_world_model.networks.transformer import Transformer, TransformerConfig
from src.world_models.iris_world_model.utils import init_weights, LossWithIntermediateLosses, configure_optimizer


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, name: str, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()

        self.name = name
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return self.name

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Dict[str, torch.Tensor], tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            observation = batch['rgb'][:, :, 0, :, :, :]  # Take the first frame of the frame stack
            observation = einops.rearrange(observation, 'b t h w c -> b t c h w')
            observation = observation / 255.0
            obs_tokens = tokenizer.encode(observation, should_preprocess=False).tokens  # (BL, K)

        act_tokens = einops.rearrange(batch['actions'], 'b l -> b l 1')
        tokens = einops.rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        outputs = self(tokens)

        ends = torch.logical_or(batch['terminateds'], batch['truncateds'])
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], ends)

        logits_observations = einops.rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(einops.rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(einops.rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO: why is this here? Can't we have a sequence of two dones close to each other
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done

        labels_observations = einops.rearrange(obs_tokens, 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.long()
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
