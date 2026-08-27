import torch
import torch.nn as nn


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6

