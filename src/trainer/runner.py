# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import time
import numpy as np
import torch
import gymnasium as gym

import random


from omegaconf import OmegaConf
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from wandb.plot import visualize

from src.agent.actor_critic import Agent
from src.environments.utils import make_collector_env
from src.environments.dataset import Dataset
from src.environments.training_atari_env import AtariGymEnv, AtariWMEnv
from src.visualization.plotting import plot_images_grid
from src.visualization.videos import create_video_from_images


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
            [make_collector_env(cfg.env_id) for _ in range(self.cfg.num_collector_envs)],
        )
        self.collector_envs.reset()

        self.train_envs = self.create_training_envs()
        self.eval_envs = self.create_eval_envs()

        self.agent = Agent(action_space=self.collector_envs.single_action_space.n).to(self.device)

        if self.cfg.agent.load_checkpoint_path is not None:
            print(f"Loading agent model from {self.cfg.agent.load_checkpoint_path}")
            self.agent.load_model(self.cfg.agent.load_checkpoint_path)

        self.agent_optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.training.learning_rate, eps=1e-5)

        self.total_iterations = self.cfg.total_steps // self.cfg.steps_per_collection
        self.current_iteration = 0
        self.dataset_buffer: Dataset = None


    def run(self):

        while self.current_iteration < self.total_iterations:

            if self.current_iteration % self.cfg.training.save_every == 0:
                self.save_checkpoint()

            # if self.current_iteration == 0:
            #     new_dataset = self.collect_data()
            #     self.add_to_dataset(new_dataset)

            # for world_model in self.world_models:
            #
            #     # Train the world model
            #     wm_train_metrics = world_model.train_model(dataset=self.dataset_buffer)
            #
            #     # Evaluate the world model
            #     if self.current_iteration % self.cfg.eval.frequency == 0:
            #         world_model.evaluate_encoder(new_dataset)

            #self.evaluate_agent(envs=self.eval_envs)
            self.train_agent_in_env(envs=self.train_envs)

            self.current_iteration += 1

        self.writer.close()


    def create_training_envs(self):
        if self.cfg.training.env_type == "gym":
            envs = [AtariGymEnv(env_id=self.cfg.env_id, num_envs=8, env_type="training")]
        elif self.cfg.training.env_type == "wm":
            envs = []
            for wm in self.cfg.world_models:
                env = AtariWMEnv(self.cfg.env_id, wm, device=self.device)
                envs.append(env)
        else:
            raise ValueError(f"Unknown training environment: {self.cfg.training.env_type}")
        return envs

    def create_eval_envs(self):
        if self.cfg.eval.env_type == "gym":
            envs = [AtariGymEnv(env_id=self.cfg.env_id, num_envs=8, env_type="evaluation")]
        elif self.cfg.eval.env_type == "wm":
            envs = []
            for wm in self.cfg.world_models:
                env = AtariWMEnv(self.cfg.env_id, wm, device=self.device)
                envs.append(env)
        else:
            raise ValueError(f"Unknown evaluation environment: {self.cfg.eval.env_type}")
        return envs

    def add_to_dataset(self, new_dataset: Dataset):
        if self.dataset_buffer is None:
            self.dataset_buffer = new_dataset
        else:
            self.dataset_buffer.append(new_dataset)

    def save_checkpoint(self):
        print("Saving")
        if self.cfg.training.env_type == "gym":
            self.agent.save_model(f"{self.cfg.save_path}/{self.cfg.env_id}_agent_last.pt")

        if self.cfg.training.env_type == "wm":
            self.agent.save_model(f"{self.cfg.save_path}/{self.cfg.env_id}_agent_in_wm_last.pt")
            for world_model in self.train_envs:
                world_model.save_checkpoint(
                    f"{self.cfg.save_path}/{self.cfg.env_id}_world_model-{world_model.env.name}_last.pt")

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


    def collect_data(self):

        frames_per_env = self.cfg.steps_per_collection // self.cfg.num_collector_envs
        frame_stack = 4

        greyscale_shape = self.collector_envs.single_observation_space["greyscale"].shape
        rgb_shape = self.collector_envs.single_observation_space["rgb"].shape

        grey_obs_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack + 1, *greyscale_shape[:-1]), dtype=np.uint8)
        rgb_obs_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack + 1, *rgb_shape), dtype=np.uint8)
        actions_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack), dtype=np.uint8)

        logprobs_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack), dtype=np.float32)
        values_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack), dtype=np.float32)

        rewards_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack), dtype=np.float32)
        terminateds_buffer = np.ones(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack + 1), dtype=np.bool_)
        truncateds_buffer = np.ones(shape=(self.cfg.num_collector_envs, frames_per_env + frame_stack + 1), dtype=np.bool_)

        # obs, _ = self.collector_envs.reset()
        obs = self.collector_envs._observations

        next_grey_obs = obs['greyscale'].squeeze(-1)
        next_rgb_obs = obs['rgb']
        next_terminateds = np.zeros(shape=self.cfg.num_collector_envs)
        next_truncateds = np.zeros(shape=self.cfg.num_collector_envs)

        total_rewards = []
        rewards_per_episode = np.zeros(self.cfg.num_collector_envs)
        for step in tqdm(range(frames_per_env + frame_stack), desc="Generating dataset"):

            # Make a random action for the first few steps
            if step < frame_stack:
                action = np.random.randint(0, self.collector_envs.single_action_space.n, self.cfg.num_collector_envs)
                logprob = np.ones(shape=self.cfg.num_collector_envs)
                value = np.zeros(shape=self.cfg.num_collector_envs)
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

        stacked_greyscale_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env, frame_stack, *greyscale_shape[:-1]), dtype=np.uint8)
        stacked_rgb_buffer = np.empty(shape=(self.cfg.num_collector_envs, frames_per_env, frame_stack, *rgb_shape), dtype=np.uint8)
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


    def evaluate_agent(self, envs):

        for env in envs:
            obs, _ = env.reset()

            step = 0
            finished_episodes = 0
            total_rewards = []
            total_episode_lengths = []
            rewards_per_episode = np.zeros(shape=(env.num_envs,))
            episode_lengths = np.zeros(shape=(env.num_envs,))
            pbar = tqdm(total=self.cfg.eval.total_episodes, desc="Evaluating agent")

            record_observations = []
            record_rewards = []
            while finished_episodes < self.cfg.eval.total_episodes:
                if step >= self.cfg.eval.max_steps:
                    print(f"Max frames reached, evaluated {finished_episodes} episodes.")
                    break

                if step % 100 == 0:
                    print(step)

                obs = torch.Tensor(obs).to(self.device)
                action, logprob, _, value = self.agent.get_action_and_value(obs)
                action = action.detach().cpu().numpy()

                obs, rewards, dones, infos = env.step(action)
                rewards_per_episode += rewards

                if self.cfg.eval.env_type == "wm":
                    record_observations.append(obs[0][0].detach().cpu().numpy())
                else:
                    record_observations.append(obs[0][0])
                record_rewards.append(rewards_per_episode[0])

                for n, done in enumerate(dones):
                    if done:
                        total_rewards.append(rewards_per_episode[n])
                        total_episode_lengths.append(episode_lengths[n])
                        rewards_per_episode[n] = 0
                        episode_lengths[n] = 0
                        finished_episodes += 1
                        pbar.update(1)

                episode_lengths += 1
                step += 1

            if self.cfg.eval.save_video:
                create_video_from_images(
                    images=np.array(record_observations),  # Your numpy array of images
                    episode_returns=np.array(record_rewards),
                    output_path='./recordings/my_video.mp4',
                    fps=15,
                    scale_factor=6  # Makes 84x84 -> 504x504 for better visibility
                )

            self.writer.add_scalar(f"eval/mean_episode_reward", np.mean(total_rewards), self.current_iteration)
            self.writer.add_scalar(f"eval/min_reward", np.min(total_rewards), self.current_iteration)
            self.writer.add_scalar(f"eval/max_reward", np.max(total_rewards), self.current_iteration)
            self.writer.add_scalar(f"eval/std_reward", np.std(total_rewards), self.current_iteration)
            self.writer.add_scalar(f"eval/mean_episode_length", np.mean(total_episode_lengths), self.current_iteration)
            self.writer.add_scalar(f"eval/min_episode_length", np.min(total_episode_lengths), self.current_iteration)
            self.writer.add_scalar(f"eval/max_episode_length", np.max(total_episode_lengths), self.current_iteration)
            self.writer.add_scalar(f"eval/std_episode_length", np.std(total_episode_lengths), self.current_iteration)


    def train_agent_in_env(self, envs):

        # TRY NOT TO MODIFY: seeding
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic

        for env in envs:

            batch_size = int(env.num_envs * self.cfg.training.num_steps)
            minibatch_size = int(batch_size // self.cfg.agent.num_minibatches)
            num_iterations = self.cfg.training.total_timesteps // batch_size

            initial_obs, _ = env.reset()
            # ALGO Logic: Storage setup
            obs = torch.zeros(self.cfg.training.num_steps, *initial_obs.shape).to(self.device)
            actions = torch.zeros((self.cfg.training.num_steps, env.num_envs)).to(self.device)
            logprobs = torch.zeros((self.cfg.training.num_steps, env.num_envs)).to(self.device)
            rewards = torch.zeros((self.cfg.training.num_steps, env.num_envs)).to(self.device)
            dones = torch.zeros((self.cfg.training.num_steps, env.num_envs)).to(self.device)
            values = torch.zeros((self.cfg.training.num_steps, env.num_envs)).to(self.device)

            # TRY NOT TO MODIFY: start the game
            global_step = 0
            start_time = time.time()
            next_obs = torch.Tensor(initial_obs).to(self.device)
            next_done = torch.zeros(env.num_envs).to(self.device)

            total_return = np.zeros(shape=(env.num_envs,), dtype=np.float32)
            total_length = np.zeros(shape=(env.num_envs,), dtype=np.float32)

            for iteration in tqdm(range(1, num_iterations + 1), desc=f"Training agent in"):
                # Annealing the rate if instructed to do so.
                if self.cfg.training.anneal_lr:
                    frac = 1.0 - (self.current_iteration - 1.0) / self.total_iterations
                    lrnow = frac * self.cfg.training.learning_rate
                    self.agent_optimizer.param_groups[0]["lr"] = lrnow

                # Collect agent training data
                for step in range(0, self.cfg.training.num_steps):
                    global_step += env.num_envs
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

                    for i in range(env.num_envs):
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
                    for t in reversed(range(self.cfg.training.num_steps)):
                        if t == self.cfg.training.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.cfg.agent.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.cfg.agent.gamma * self.cfg.agent.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values



                #### ~DEBUG PLOTTING

                # x = obs[:, 0].cpu().detach().numpy()
                #
                # def plot_multiple_images(x, n_images=16, figsize=(12, 8), channel=0):
                #     """Plot a grid of images showing specified channel"""
                #     n_cols = 4
                #     n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
                #
                #     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
                #     fig.suptitle(f'First {n_images} Images - Channel {channel}', fontsize=14)
                #
                #     # Flatten axes array for easier indexing
                #     if n_rows == 1:
                #         axes = [axes] if n_cols == 1 else axes
                #     else:
                #         axes = axes.flatten()
                #
                #     for i in range(n_images):
                #         row, col = divmod(i, n_cols)
                #         axes[i].imshow(x[i, channel], cmap='gray')
                #         axes[i].set_title(f'Img {i}')
                #         axes[i].axis('off')
                #
                #     # Hide empty subplots
                #     for i in range(n_images, len(axes)):
                #         axes[i].axis('off')
                #
                #     plt.tight_layout()
                #     plt.show()
                #
                # plot_multiple_images(x)
                #### ~DEBUG_PLOTTING
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
                #print("SPS:", int(global_step / (time.time() - start_time)))
                self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


