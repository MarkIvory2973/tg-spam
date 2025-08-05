import os
import torch
import unicodedata
from torch import nn, optim
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from core import datasets, models, metrics

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")