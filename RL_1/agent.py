import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(self.device)

    def act(self, state):
        model = self.model
        model.eval()

        if isinstance(state, torch.Tensor):
            state_tensor = state
        elif isinstance(state, list):
            state_tensor = torch.tensor(state)
        else:
            state_tensor = torch.from_numpy(state)

        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            preds = model(state_tensor)
            return torch.argmax(preds).item()
