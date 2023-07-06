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

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
#DEVICE = "cuda"
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 2048*2#1024#128
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1000000

EXPL_CLIP = 0.5
SIGMA = 2

torch.manual_seed(7)
np.random.seed(3)

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

    def forward(self, state):
        return self.model(state)


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


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=CRITIC_LR)
        #self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)  # ???
        #self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)  # ???

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = deque(maxlen=200000)

    def update(self, transition):
        self.replay_buffer.append(transition)
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
            with torch.no_grad():
                next_action = torch.clip(self.target_actor(next_state), -1, 1)
                #next_action = next_action + torch.clip(SIGMA * torch.randn_like(next_action), -EXPL_CLIP, EXPL_CLIP)
                # next_action = torch.clip(next_action + SIGMA * torch.clip(torch.randn_like(next_action),
                #                                                           -EXPL_CLIP, EXPL_CLIP), -1, +1)
                next_action = torch.clip(next_action + 0.3 * torch.randn_like(next_action), -1, +1)
                # if np.all(done == [1]*):
                #     q_t = reward
                # else:
                #     q_t = reward + GAMMA * torch.minimum(self.target_critic_1(next_state, next_action), self.target_critic_2(next_state, next_action))
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
            loss_actor = -torch.mean(self.critic_1(state, self.actor(state)))
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # soft upd
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


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make(ENV_NAME)
    test_enenv = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    eps = 0.2

    max_r = 0
    for i in range(TRANSITIONS):
        #print(f'step{i+1}')
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + eps * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            td3.save()
            if np.mean(rewards) > max_r:
                max_r = np.mean(rewards)
                torch.save(td3.actor.state_dict(), "best.pt")
