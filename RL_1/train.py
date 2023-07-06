from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque, OrderedDict
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

BUFFER_SIZE = 1000

np.random.seed(19)


class NNet(nn.Module):
    def __init__(self, input_size, num_layers, hidden_sizes, activations, output_size):
        super(NNet, self).__init__()

        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers

        #flat = ('flat', nn.Flatten())
        in_to_hid = ('in2hid', nn.Linear(input_size, hidden_sizes[0]))

        hid_ = [[
            (f'act_{i + 1}', activations[i]),
            (f'hid_{i + 1}', nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        ] for i in range(num_layers - 1)]
        hid = []
        for el in hid_:
            hid.extend(el)

        head = [
            (f'act_{num_layers}', activations[-1]),
            ('hid2out', nn.Linear(hidden_sizes[-1], output_size))
        ]

        #self.net = [flat, in_to_hid, *hid, *head]
        self.net = [in_to_hid, *hid, *head]
        self.net = nn.Sequential(OrderedDict(self.net))

    def forward(self, s):
        a = self.net(s)
        return a


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = NNet(input_size=state_dim,
                          num_layers=2,
                          hidden_sizes=16,
                          activations=nn.ReLU(),
                          output_size=action_dim).to(self.device)
        self.target_model = NNet(input_size=state_dim,
                                 num_layers=2,
                                 hidden_sizes=16,
                                 activations=nn.ReLU(),
                                 output_size=action_dim).to(self.device)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.gamma = GAMMA
        self.optimizer = Adam(self.model.parameters(), LEARNING_RATE)
        self.loss = F.mse_loss

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        idx = np.random.randint(0, BUFFER_SIZE, size=BATCH_SIZE)
        batch = [self.replay_buffer[ind] for ind in idx]


        # q = list(zip(*batch))
        # qq = q[0]
        # qqq = np.array(qq)
        # qqq = torch.from_numpy(qqq).to(self.device)

        grouped_batch = [torch.from_numpy(np.array(el)).to(self.device) for el in zip(*batch)]
        return grouped_batch

    def train_step(self, batch):
        # Use batch to update DQN's network.
        # self.model.train()
        states, actions, next_states, rewards, dones = batch

        target_model = self.target_model
        target_model.eval()
        with torch.no_grad():
            preds_target = target_model(next_states)
            preds_target = torch.amax(preds_target, dim=1)
            preds_target[dones] = 0
            preds_target *= self.gamma
            preds_target += rewards

        model = self.model
        model.train()
        self.optimizer.zero_grad()

        preds = model(states)
        preds = preds[torch.arange(preds.shape[0]), actions.long()]

        loss = self.loss(preds, preds_target)
        loss.backward()
        self.optimizer.step()

        #self.update_target_network()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        #state = np.array(state)

        model = self.model.eval()
        with torch.no_grad():
            state = torch.from_numpy(np.array(state)).to(self.device)
            preds = model(state)
            action = torch.argmax(preds)

        return action.item()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        state = state[0]
        total_reward = 0.

        while not done:
            # state, reward, done, _, _ = env.step(agent.act(state))
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    state = state[0]

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        #next_state, reward, done, _, _ = env.step(action)


        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()
        if len(state) == 2:
            state = state[0]

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        #next_state, reward, done, _, _ = env.step(action)
        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()
        if len(state) == 2:
            state = state[0]

        print(f'step {i}')

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
