import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
import pickle

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 1024
ITERATIONS = 1000000

l_kld = 1
l_cql = 1
EXPL_CLIP = 0.3
SIGMA = 2
ENTROPY_COEF = 1e-2

def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))#.to(DEVICE)

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand(mu.size())
        distrib = torch.distributions.Normal(mu, sigma)
        probs = torch.exp(torch.sum(distrib.log_prob(action), -1))
        return probs, distrib

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand(mu.size())
        distrib = torch.distributions.Normal(mu, sigma)
        not_tr_act = distrib.sample()
        action = torch.tanh(not_tr_act)
        return action, not_tr_act, distrib
    #def forward(self, state):
    #    return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class Algo:
    def __init__(self, state_dim, action_dim, data):
        # TODO: You can modify anything here
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = list(data)

    def update(self):
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            # Update critic
            # TODO: Implement critic update
            with torch.no_grad():
                #next_action = self.target_actor(next_state)
                next_action, _, _ = self.target_actor.act(next_state)
                next_action = torch.clip(
                    next_action + torch.clip(torch.randn_like(next_action), -EXPL_CLIP, EXPL_CLIP), -1, +1)
                q_t = reward + GAMMA * (1 - done) * torch.minimum(self.target_critic_1(next_state, next_action),
                                                                  self.target_critic_2(next_state, next_action))

            loss_critic1 = F.mse_loss(self.critic_1(state, action), q_t)
            loss_critic2 = F.mse_loss(self.critic_2(state, action), q_t)

            self.critic_1_optim.zero_grad()
            loss_critic1.backward()
            self.critic_1_optim.step()

            self.critic_2_optim.zero_grad()
            loss_critic2.backward()
            self.critic_2_optim.step()

            # Update actor
            # TODO: Implement actor update
            cur_actions, _, _ = self.actor.act(state)
            q_pi = self.critic_1(state, cur_actions)
            loss_actor = -torch.mean(q_pi)

            # policy constraints
            _, distrib_cur = self.actor.compute_proba(state, cur_actions)
            _, distrib_exp = self.actor.compute_proba(state, action)
            loss_kld = torch.distributions.kl.kl_divergence(distrib_cur, distrib_exp)
            # expert_actions = action
            # current_dist = torch.distributions.Normal(cur_actions, 1)
            # expert_dist = torch.distributions.Normal(expert_actions, 1)
            # loss_kld = torch.distributions.kl.kl_divergence(current_dist, expert_dist)
            #

            # cql
            with torch.no_grad():
                q_exp = torch.minimum(self.target_critic_1(state, action),
                                      self.target_critic_2(state, action))
            loss_cql = q_pi - q_exp
            #

            loss_actor += l_kld * torch.mean(loss_kld)
            loss_actor += l_cql * torch.mean(loss_cql)
            loss_actor -= ENTROPY_COEF * distrib_cur.entropy().mean()

            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        #torch.save(self.actor, "agent.pkl")
        torch.save(self.actor.state_dict(), "agent.pt")


if __name__ == "__main__":
    with open("./offline_data.pkl", "rb") as f:
        data = pickle.load(f)

    algo = Algo(state_dim=32, action_dim=8, data=data)

    for i in range(ITERATIONS):
        if i % 10000 == 0:
            print(f'step {i}')
        if i % 100000 == 0:
            torch.save(algo.actor.state_dict(), "agent.pt")
        steps = 0
        algo.update()
    torch.save(algo.actor.state_dict(), "agent.pt")
