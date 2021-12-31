import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules import loss
import torch.nn.functional as F

'''
train functions for quadratic models
'''




# region QAE
def group_parameters(m):
    group_r = list(map(lambda x: x[1], list(filter(lambda kv: '_r' in kv[0], m.named_parameters()))))
    group_g = list(map(lambda x: x[1], list(filter(lambda kv: '_g' in kv[0], m.named_parameters()))))
    group_b = list(map(lambda x: x[1], list(filter(lambda kv: '_b' in kv[0], m.named_parameters()))))
    return group_r, group_g, group_b
# endregion
