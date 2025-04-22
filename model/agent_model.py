import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseAgent(nn.Module):
    def __init__(self, args):
        super(BaseAgent, self).__init__()
        self.args = args
        self.input_shape = args.obs_shape + args.n_agents + 1
        self.fc1 = nn.Linear(self.input_shape, args.agent_hidden_dim)
        self.fc2 = nn.Linear(args.agent_hidden_dim, args.agent_hidden_dim)
        self.fc3 = nn.Linear(args.agent_hidden_dim, args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        q = self.fc3(x)
        return q
