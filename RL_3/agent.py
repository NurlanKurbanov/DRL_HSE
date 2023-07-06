import random
import numpy as np
import os
import torch
from .train import Actor

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Agent:
    def __init__(self):
        self.model = Actor(28, 8).to(DEVICE)
        #self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))

    def act(self, state):
        model = self.model.eval()
        with torch.no_grad():
            state = torch.tensor(np.array(state)).unsqueeze(0).float().to(DEVICE)
            #a = model.act(state)[0].flatten().cpu().numpy()
            a = model(state)[0].flatten().cpu().numpy()
        return a  # TODO

    def reset(self):
        pass


# import pybullet_envs
# from gym import make
# a = Agent()
# env = make("AntBulletEnv-v0")
# state = env.reset()
# while 1:
#     a = a.act(state)