import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sincos_pe(hidden_size, max_len=1000):  
    pe = torch.zeros(max_len, hidden_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe
