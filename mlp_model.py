#%%
'''
with help from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
'''
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error 
import config
#%%


class MLP_model(nn.Module):

## init the superclass
  def __init__(self):
    super().__init__()
    # input first flows through the first layer, followed by the second, followed by..
    self.layers = nn.Sequential(
      nn.Linear(config.l0, config.l1),
      nn.ReLU(),
      nn.Linear(config.l1, config.l2),
      nn.ReLU(),
      nn.Linear(config.l2, 1)
    ).cuda()

    
  def forward(self, x):
    '''
      Forward pass: feed the input data through the model (self.layers) and return the result
    '''
    return self.layers(x)
