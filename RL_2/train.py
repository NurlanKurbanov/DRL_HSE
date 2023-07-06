import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

torch.manual_seed(7)
np.random.seed(3)

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.97#0.95
GAMMA = 0.99 #0.99

ACTOR_LR = 2.5e-4#2e-4
CRITIC_LR = 2e-4#1e-4

CLIP = 0.2
ENTROPY_COEF = 1e-3#1e-2
BATCHES_PER_UPDATE = 64#64
BATCH_SIZE = 1024

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1000

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')

def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        ).to(DEVICE)
        self.sigma = nn.Parameter(torch.zeros(action_dim)).to(DEVICE)

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand(mu.size())
        distrib = Normal(mu, sigma)
        probs = torch.exp(torch.sum(distrib.log_prob(action), -1))
        return probs, distrib

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand(mu.size())
        distrib = Normal(mu, sigma)
        not_tr_act = distrib.sample()
        action = torch.tanh(not_tr_act)
        return action, not_tr_act, distrib


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        ).to(DEVICE)

    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx]).float().to(DEVICE)
            a = torch.tensor(action[idx]).float().to(DEVICE)
            op = torch.tensor(old_prob[idx]).float().to(DEVICE)  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(DEVICE)  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float().to(DEVICE)  # Estimated by generalized advantage estimation

            # TODO: Update actor here
            prob_cur, distrib_cur = self.actor.compute_proba(s, a)
            imp_sampling_coef = prob_cur / op
            loss_actor = torch.minimum(imp_sampling_coef * adv,
                                       torch.clip(imp_sampling_coef, 1 - CLIP, 1 + CLIP) * adv)
            loss_actor = -torch.mean(loss_actor)
            loss_actor -= ENTROPY_COEF * distrib_cur.entropy().mean()

            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # TODO: Update critic here
            val_cur = self.critic.get_value(s)
            val_cur = val_cur.flatten()
            loss_critic = F.mse_loss(val_cur, v)

            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()


    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(DEVICE)
            value = self.critic.get_value(state)
        #return value.cpu().item()
        return value.to(DEVICE).item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(DEVICE)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()
        #return action.to(DEVICE).numpy()[0], pure_action.to(DEVICE).numpy()[0], prob.to(DEVICE).item()

    def save(self):
        #torch.save(self.actor, "agent.pkl")
        torch.save(self.actor.state_dict(), "agent.pt")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    best_rew = 0
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)

        print(f'iteration: {i+1}')

        if (i + 1) % (ITERATIONS // 100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            cur_rew_mean = np.mean(rewards)
            print(
                f"Step: {i + 1}, Reward mean: {cur_rew_mean}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            if cur_rew_mean > best_rew:
                best_rew = cur_rew_mean
                ppo.save()