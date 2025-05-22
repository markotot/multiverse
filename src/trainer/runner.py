# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import time
import numpy as np
import torch
import gymnasium as gym

import random

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from src.agent.actor_critic import Agent
from src.environments import data_collector_env
from src.environments.dataset import Dataset
from src.environments.training_gym_env import AtariGymEnv
from src.world_models.iris_world_model.networks.world_model import WorldModel
from src.world_models.iris_world_model.iris_env import IrisEnv

from src.visualization.plotting import plot_images
from src.world_models.iris_world_model.utils import extract_state_dict


class Runner:
    def __init__(self, cfg):

        self.cfg = cfg
        self.device = self.setup_seed_and_device()

        self.run_name = f"{self.cfg.env_id}__{self.cfg.exp_name}__{self.cfg.seed}__{int(time.time())}"
        if self.cfg.use_wandb:
            self.setup_wandb()
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.cfg).items()])),
            )

        self.collector_envs = gym.vector.SyncVectorEnv(
            [data_collector_env.make_env(cfg.env_id, i, self.cfg.capture_video) for i in range(self.cfg.num_envs)],
        )
        self.collector_envs.reset()

        self.eval_envs = gym.vector.SyncVectorEnv(
            [data_collector_env.make_env(self.cfg.env_id, i, self.cfg.capture_video) for i in range(self.cfg.num_envs)],
        )
        self.eval_envs.reset()

        self.agent = Agent(action_space=self.collector_envs.single_action_space.n).to(self.device)
        self.world_models : list[IrisEnv] = self.initialize_world_models(self.cfg.world_models)


        self.agent_optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

        self.total_iterations = self.cfg.total_steps // self.cfg.steps_per_collection
        self.current_iteration = 0

        self.dataset_buffer: Dataset = None


    def add_to_dataset(self, new_dataset: Dataset):
        if self.dataset_buffer is None:
            self.dataset_buffer = new_dataset
        else:
            self.dataset_buffer.append(new_dataset)

    def setup_seed_and_device(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic
        device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")
        return device

    def setup_wandb(self):
            wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                sync_tensorboard=True,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
                dir="./runs/wandb"
            )

    def initialize_world_models(self, world_models_cfg) -> list[IrisEnv]:
        models = []
        for wm in world_models_cfg:
            # Load the world model configuration
            if wm.type == "iris":

                wm_transformer_parts = wm.transformer.split("/")
                wm_tokenizer_parts = wm.tokenizer.split("/")

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
                world_model = WorldModel(name=wm.name,
                                         obs_vocab_size=tokenizer.vocab_size,
                                         act_vocab_size=self.collector_envs.single_action_space.n,
                                         config=transformer_cfg,
                                        )


                iris_world_model = IrisEnv(name=wm.name,
                                           device=self.device,
                                           tokenizer=tokenizer,
                                           world_model=world_model,
                                           training_cfg=iris_cfg.training)
                if wm.load_checkpoint:
                    self.load_wm_checkpoint(iris_world_model, "./checkpoint/iris/breakout/Breakout.pt")


                iris_world_model.to(self.device)

            else:
                raise NotImplementedError

            models.append(iris_world_model)

        return models

    def collect_data(self):

        # Evaluation loop
        frames_per_env = self.cfg.steps_per_collection // self.cfg.num_envs
        frame_stack = 4

        greyscale_shape = self.collector_envs.single_observation_space["greyscale"].shape
        rgb_shape = self.collector_envs.single_observation_space["rgb"].shape

        grey_obs_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack + 1, *greyscale_shape[:-1]), dtype=np.uint8)
        rgb_obs_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack + 1, *rgb_shape), dtype=np.uint8)
        actions_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack), dtype=np.uint8)

        logprobs_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack), dtype=np.float32)
        values_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack), dtype=np.float32)

        rewards_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env + frame_stack), dtype=np.float32)
        terminateds_buffer = np.ones(shape=(self.cfg.num_envs, frames_per_env + frame_stack + 1), dtype=np.bool_)
        truncateds_buffer = np.ones(shape=(self.cfg.num_envs, frames_per_env + frame_stack + 1), dtype=np.bool_)

        # obs, _ = self.collector_envs.reset()
        obs = self.collector_envs._observations

        next_grey_obs = obs['greyscale'].squeeze(-1)
        next_rgb_obs = obs['rgb']
        next_terminateds = np.zeros(shape=self.cfg.num_envs)
        next_truncateds = np.zeros(shape=self.cfg.num_envs)

        total_rewards = []
        rewards_per_episode = np.zeros(self.cfg.num_envs)
        for step in tqdm(range(frames_per_env + frame_stack), desc="Generating dataset"):

            # Make a random action for the first few steps
            if step < frame_stack:
                action = np.random.randint(0, self.collector_envs.single_action_space.n, self.cfg.num_envs)
                logprob = np.ones(shape=self.cfg.num_envs)
                value = np.zeros(shape=self.cfg.num_envs)
            else:
                obs_stack = grey_obs_buffer[:, step - frame_stack:step, :, :]
                obs_stack = torch.from_numpy(obs_stack).to(self.device)
                action, logprob, _, value = self.agent.get_action_and_value(obs_stack)
                action = action.detach().cpu().numpy()
                logprob = logprob.detach().cpu().numpy()
                value = value.flatten().detach().cpu().numpy()

            grey_obs_buffer[:, step, :, :] = next_grey_obs
            rgb_obs_buffer[:, step, :, :, :] = next_rgb_obs
            terminateds_buffer[:, step] = next_terminateds
            truncateds_buffer[:, step] = next_truncateds
            actions_buffer[:, step] = action
            logprobs_buffer[:, step] = logprob
            values_buffer[:, step] = value

            obs, rewards, next_terminateds, next_truncateds, infos = self.collector_envs.step(action)

            rewards_buffer[:, step] = rewards
            next_grey_obs = obs['greyscale'].squeeze(-1)
            next_rgb_obs = obs['rgb']

            dones = np.logical_or(next_terminateds, next_truncateds)
            rewards_per_episode += rewards
            for i, done in enumerate(dones):
                if done:
                    total_rewards.append(rewards_per_episode[i])
                    rewards_per_episode[i] = 0

        stacked_greyscale_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env, frame_stack, *greyscale_shape[:-1]), dtype=np.uint8)
        stacked_rgb_buffer = np.empty(shape=(self.cfg.num_envs, frames_per_env, frame_stack, *rgb_shape), dtype=np.uint8)
        for i in range(frames_per_env):
            stacked_greyscale_buffer[:, i] = grey_obs_buffer[:, i:i + frame_stack, :, :]
            stacked_rgb_buffer[:, i] = rgb_obs_buffer[:, i:i + frame_stack, :, :, :]

        stacked_greyscale_buffer[:, -1] = grey_obs_buffer[:, frames_per_env: frames_per_env + frame_stack, :, :]
        stacked_rgb_buffer[:, -1] = rgb_obs_buffer[:, frames_per_env: frames_per_env + frame_stack, :, :, :]
        terminateds_buffer[:, -1] = next_terminateds
        truncateds_buffer[:, -1] = next_truncateds


        dataset = Dataset(
                        greyscale_buffer=stacked_greyscale_buffer,
                        rgb_buffer=stacked_rgb_buffer,
                        action_buffer=actions_buffer[:, frame_stack:],
                        reward_buffer=rewards_buffer[:, frame_stack:],
                        terminated_buffer=terminateds_buffer[:, frame_stack:],
                        truncated_buffer=truncateds_buffer[:, frame_stack:],
                        logprobs_buffer=logprobs_buffer[:, frame_stack:],
                        value_buffer=values_buffer[:, frame_stack:],
        )

        return dataset

    def evaluate_agent(self):

        rgb_shape = (64, 64, 3)
        greyscale_shape = (84, 84, 1)

        # Evaluation loop
        initial_steps = 4
        frame_stack = 4

        grey_obs_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps + 1, *greyscale_shape[:-1]), dtype=np.uint8)
        rgb_obs_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps + 1, *rgb_shape), dtype=np.uint8)
        actions_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps), dtype=np.uint8)
        rewards_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps), dtype=np.float32)
        terminateds_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps), dtype=np.bool_)
        truncateds_buffer = np.empty(shape=(self.cfg.num_envs, self.cfg.max_eval_steps), dtype=np.bool_)

        obs, _ = self.eval_envs.reset()
        grey_obs_buffer[:, 0, :, :] = obs['greyscale'].squeeze(-1)
        rgb_obs_buffer[:, 0, :, :, :] = obs['rgb']

        total_rewards = []
        rewards_per_episode = np.zeros(self.cfg.num_envs)

        pbar = tqdm(total=self.cfg.eval_episodes, desc="Evaluating agent")

        finished_episodes = 0
        step = 0
        while finished_episodes < self.cfg.eval_episodes:

            if step >= self.cfg.max_eval_steps:
                print(f"Max frames reached, evaluated {finished_episodes} episodes.")
                break
            # Make a random action for the first few steps
            if step < initial_steps:
                action = np.random.randint(0, 4, self.cfg.num_envs)
            else:
                obs_stack = grey_obs_buffer[:, step - frame_stack:step, :, :]  # Assume frame_stack < initial_steps
                obs_stack = torch.from_numpy(obs_stack).to(self.device)
                action, logprob, _, value = self.agent.get_action_and_value(obs_stack)
                action = action.detach().cpu().numpy()

            obs, rewards, terminateds, truncateds, infos = self.eval_envs.step(action)
            dones = np.logical_or(terminateds, truncateds)
            rewards_per_episode += rewards
            for n, done in enumerate(dones):
                if done:
                    total_rewards.append(rewards_per_episode[n])
                    rewards_per_episode[n] = 0
                    finished_episodes += 1
                    pbar.update(1)

            grey_obs_buffer[:, step + 1, :, :] = obs['greyscale'].squeeze(-1)
            rgb_obs_buffer[:, step + 1, :, :, :] = obs['rgb']
            actions_buffer[:, step] = action
            rewards_buffer[:, step] = rewards
            terminateds_buffer[:, step] = terminateds
            truncateds_buffer[:, step] = truncateds

            step += 1

        pbar.close()
        return np.array(total_rewards)

    def train_agent_in_env(self, env, dataset):

        batch_size = int(self.cfg.num_envs * self.cfg.num_steps)
        minibatch_size = int(batch_size // self.cfg.agent.num_minibatches)
        num_iterations = self.cfg.total_timesteps // batch_size

        # TRY NOT TO MODIFY: seeding
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic

        # Uncomment if using world model
        # envs = self.world_models[0] # TODO: Uncomment if using world model
        # initial_obs, info = envs.reset(dataset=dataset)
        # initial_obs = next_obs['greyscale'].squeeze(-1)

        initial_obs, _ = env.reset()


        # ALGO Logic: Storage setup
        obs = torch.zeros(self.cfg.num_steps, *initial_obs.shape).to(self.device)
        actions = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        logprobs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        rewards = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        dones = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)
        values = torch.zeros((self.cfg.num_steps, self.cfg.num_envs)).to(self.device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(initial_obs).to(self.device)
        next_done = torch.zeros(self.cfg.num_envs).to(self.device)

        total_return = np.zeros(shape=(self.cfg.num_envs,), dtype=np.float32)
        total_length = np.zeros(shape=(self.cfg.num_envs,), dtype=np.float32)

        for iteration in tqdm(range(1, num_iterations + 1), desc="Training agent in WM"):
            # Annealing the rate if instructed to do so.
            if self.cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.cfg.learning_rate
                self.agent_optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.cfg.num_steps):
                global_step += self.cfg.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, infos = env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                total_return += reward
                total_length += 1

                for i in range(self.cfg.num_envs):
                    if next_done[i]:
                        self.writer.add_scalar("charts/episodic_return", total_return[i], global_step)
                        self.writer.add_scalar("charts/episodic_length", total_length[i], global_step)
                        total_return[int(i)] = 0
                        total_length[int(i)] = 0

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.cfg.num_steps)):
                    if t == self.cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.cfg.agent.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.cfg.agent.gamma * self.cfg.agent.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(self.cfg.agent.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    b_obs_indices = b_obs[mb_inds]
                    b_act_indices = b_actions.long()[mb_inds]
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs_indices,b_act_indices)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.cfg.agent.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.cfg.agent.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.agent.clip_coef, 1 + self.cfg.agent.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.cfg.agent.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg.agent.clip_coef,
                            self.cfg.agent.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.cfg.agent.ent_coef * entropy_loss + v_loss * self.cfg.agent.vf_coef

                    self.agent_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.agent.max_grad_norm)
                    self.agent_optimizer.step()

                if self.cfg.agent.target_kl is not None and approx_kl > self.cfg.agent.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.agent_optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        #writer.close()
        print("Test")

    # def train_agent(self, dataset: Dataset) -> dict:
    #
    #     batch_size = int(self.cfg.num_envs * self.cfg.num_steps)
    #     minibatch_size = int(batch_size // self.cfg.num_minibatches)
    #     num_batches = dataset.action_buffer.shape[1] // self.cfg.num_steps
    #
    #     obs = dataset.greyscale_buffer.transpose(1, 0, *range(2, dataset.greyscale_buffer.ndim))
    #     actions = dataset.action_buffer.transpose(1, 0)
    #     rewards = dataset.reward_buffer.transpose(1, 0)
    #     terminates = dataset.terminated_buffer.transpose(1, 0)
    #     truncates = dataset.truncated_buffer.transpose(1, 0)
    #     logprobs = dataset.logprobs_buffer.transpose(1, 0)
    #     values = dataset.value_buffer.transpose(1, 0)
    #
    #     obs = torch.from_numpy(obs).to(self.device)
    #     rewards = torch.from_numpy(rewards).to(self.device)
    #     terminates = torch.from_numpy(terminates).to(self.device)
    #     truncates = torch.from_numpy(truncates).to(self.device)
    #     dones = torch.logical_or(terminates, truncates).to(self.device)
    #
    #     actions = torch.from_numpy(actions).to(self.device)
    #     logprobs = torch.from_numpy(logprobs).to(self.device)
    #     values = torch.from_numpy(values).to(self.device)
    #
    #
    #     batch_start_indices = np.arange(0, obs.shape[0], 128)
    #
    #     if self.cfg.anneal_lr:
    #         frac = 1.0 - (self.current_iteration - 1.0) / self.total_iterations
    #         new_lr = frac * self.cfg.learning_rate
    #         self.optimizer.param_groups[0]["lr"] = new_lr
    #
    #     # Do this for every batch
    #     for curr_batch in tqdm(range(0, num_batches), desc="Training batches"):
    #         # Annealing the rate if instructed to do so.
    #
    #         obs_batch = obs[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #         actions_batch = actions[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #         logprobs_batch = logprobs[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #         rewards_batch = rewards[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #         dones_batch = dones[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #         values_batch = values[batch_start_indices[curr_batch]: batch_start_indices[curr_batch + 1]]
    #
    #         with torch.no_grad():
    #             stacked_obs = obs[batch_start_indices[curr_batch+1]] # .transpose(1, 0)
    #             next_value = self.agent.get_value(stacked_obs).reshape(1, -1)
    #             advantages_batch = torch.zeros_like(rewards_batch).to(self.device)
    #             lastgaelam = 0
    #             for t in reversed(range(self.cfg.num_steps)):
    #                 if t == self.cfg.num_steps - 1:
    #                     nextnonterminal = 1.0 - dones[batch_start_indices[curr_batch+1]].to(torch.float32)
    #                     nextvalues = next_value
    #                 else:
    #                     nextnonterminal = 1.0 - dones_batch[t + 1].to(torch.float32)
    #                     nextvalues = values[t + 1]
    #                 delta = rewards_batch[t] + self.cfg.gamma * nextvalues * nextnonterminal - values_batch[t]
    #                 advantages_batch[t] = lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
    #             returns_batch = advantages_batch + values_batch
    #
    #         # flatten the batch
    #         b_obs = obs_batch.reshape(shape=(obs_batch.shape[0] * obs_batch.shape[1], *obs_batch.shape[2:]))
    #         b_logprobs = logprobs_batch.reshape(-1)
    #         b_actions = actions_batch.reshape((-1,) + self.collector_envs.single_action_space.shape)
    #         b_advantages = advantages_batch.reshape(-1)
    #         b_returns = returns_batch.reshape(-1)
    #         b_values = values_batch.reshape(-1)
    #
    #         # Optimizing the policy and value network
    #         b_inds = np.arange(batch_size)
    #         clipfracs = []
    #         for epoch in range(self.cfg.update_epochs):
    #             np.random.shuffle(b_inds)
    #             for start in range(0, batch_size, minibatch_size):
    #                 end = start + minibatch_size
    #                 mb_inds = b_inds[start:end]
    #
    #                 _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
    #                                                                               b_actions.long()[mb_inds])
    #                 logratio = newlogprob - b_logprobs[mb_inds]
    #                 ratio = logratio.exp()
    #
    #                 with torch.no_grad():
    #                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
    #                     old_approx_kl = (-logratio).mean()
    #                     approx_kl = ((ratio - 1) - logratio).mean()
    #                     clipfracs += [((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()]
    #
    #                 mb_advantages = b_advantages[mb_inds]
    #                 if self.cfg.norm_adv:
    #                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    #
    #                 # Policy loss
    #                 pg_loss1 = -mb_advantages * ratio
    #                 pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
    #                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    #
    #                 # Value loss
    #                 newvalue = newvalue.view(-1)
    #                 if self.cfg.clip_vloss:
    #                     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    #                     v_clipped = b_values[mb_inds] + torch.clamp(
    #                         newvalue - b_values[mb_inds],
    #                         -self.cfg.clip_coef,
    #                         self.cfg.clip_coef,
    #                     )
    #                     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    #                     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                     v_loss = 0.5 * v_loss_max.mean()
    #                 else:
    #                     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
    #
    #                 entropy_loss = entropy.mean()
    #                 loss = pg_loss - self.cfg.ent_coef * entropy_loss + v_loss * self.cfg.vf_coef
    #
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
    #                 self.optimizer.step()
    #
    #             if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
    #                 break
    #
    #         y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    #         var_y = np.var(y_true)
    #         explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    #
    #     metrics = {
    #         "learning_rate": self.optimizer.param_groups[0]["lr"],
    #         "value_loss": v_loss.item(),
    #         "policy_loss": pg_loss.item(),
    #         "entropy": entropy_loss.item(),
    #         "old_approx_kl": old_approx_kl.item(),
    #         "approx_kl": approx_kl.item(),
    #         "clipfrac": np.mean(clipfracs),
    #         "explained_variance": explained_var,
    #     }
    #     return metrics

    def run(self):

        while self.current_iteration < self.total_iterations:

            time.sleep(0.1) # Added to make the output pretty, remove for proper training

            if self.current_iteration == 0:
                new_dataset = self.collect_data()
                self.add_to_dataset(new_dataset)



            # for world_model in self.world_models:
            #
            #     # Train the world model
            #     # wm_train_metrics = world_model.train_model(dataset=self.dataset_buffer)
            #
            #     # Evaluate the world model
            #     if self.current_iteration % self.cfg.eval_frequency == 0:
            #         world_model.evaluate_encoder(new_dataset)


            #metrics = self.train_agent(new_dataset)
            env = AtariGymEnv(env_id=self.cfg.env_id,num_envs=self.cfg.num_envs)
            self.train_agent_in_env(env=env, dataset=self.dataset_buffer)

            total_reward = self.evaluate_agent()
            print(f"Iteration: {self.current_iteration}\t"
                  f"Mean reward: {np.mean(total_reward):.2f}\t"
                  f"Std: {np.std(total_reward):.2f}\t"
                  f"Max: {np.max(total_reward):.2f}\t"
                  f"Min: {np.min(total_reward):.2f}")

            self.current_iteration += 1

    def load_wm_checkpoint(self, wm, path_to_checkpoint):

        agent_state_dict = torch.load(path_to_checkpoint, map_location=self.device)

        tokenizer = extract_state_dict(agent_state_dict, 'tokenizer')
        wm.tokenizer.load_state_dict(tokenizer)

        world_model = extract_state_dict(agent_state_dict, 'world_model')
        wm.world_model.load_state_dict(world_model)


