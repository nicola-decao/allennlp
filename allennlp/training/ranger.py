import math
import torch
from torch.optim.optimizer import Optimizer, required
import itertools as it
from allennlp.training.lookahead import *
from allennlp.training.radam import * 


def Ranger(params, alpha=0.5, k=6, *args, **kwargs):
    radam = RAdam(params, *args, **kwargs)
    return Lookahead(radam, alpha, k)