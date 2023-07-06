import random
import numpy as np
import os
import torch
from .train import Actor

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')


class Agent:
    def __init__(self):
        self.model = Actor(22, 6).to(DEVICE)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))

    def act(self, state):
        model = self.model.eval()
        with torch.no_grad():
            state = torch.tensor(np.array(state)).unsqueeze(0).float().to(DEVICE)
            a = model.act(state)[0].flatten().cpu().numpy()
            return a

    def reset(self):
        pass
