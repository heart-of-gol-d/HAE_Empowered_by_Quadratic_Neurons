import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules import loss
import torch.nn.functional as F

'''
some train functions for different models
'''




# region QFAE

def group_parameters(m):
    group_r = list(map(lambda x: x[1], list(filter(lambda kv: '_r' in kv[0], m.named_parameters()))))
    group_g = list(map(lambda x: x[1], list(filter(lambda kv: '_g' in kv[0], m.named_parameters()))))
    group_b = list(map(lambda x: x[1], list(filter(lambda kv: '_b' in kv[0], m.named_parameters()))))
    return group_r, group_g, group_b
# endregion

# region SAE
def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2
# endregion


# region CAE
'''
This code reference from:
https://github.com/AlexPasqua/Autoencoders/blob/main/src/custom_losses.py
'''
class ContractiveLoss(loss.MSELoss):
    """
    Custom loss for contractive autoencoders.
    note: the superclass is MSELoss, simply because the base class _Loss is protected and it's not a best practice.
          there isn't a real reason between the choice of MSELoss, since the forward method is overridden completely.
    Overridden for elasticity -> it's possible to use a function as a custom loss, but having a wrapper class
    allows to do:
        criterion = ClassOfWhateverLoss()
        loss = criterion(output, target)    # this line always the same regardless of the type on loss
    """
    def __init__(self, ae, lambd: float, size_average=None, reduce=None, reduction: str = 'mean', model_name: str = 'CAE') -> None:
        super(ContractiveLoss, self).__init__(size_average, reduce, reduction)
        self.ae = ae
        self.lambd = lambd
        self.model_name = model_name
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return contractive_loss(input, target, self.lambd, self.ae, self.reduction, self.model_name)


def contractive_loss(input, target, lambd, ae, reduction: str, model_name:str):
    """
    Actual function computing the loss of a contractive autoencoder
    :param input: (Tensor)
    :param target: (Tensor)
    :param lambd: (float) regularization parameter
    :param ae: (DeepAutoencoder) the model itself, used to get it's weights
    :param reduction: (str) type of reduction {'mean' | 'sum'}
    :raises: ValueError
    :return: the loss
    """
    term1 = (input - target) ** 2
    if model_name == 'CAE':
        enc_weights = [ae.encoder[i].weight for i in reversed(range(0, len(ae.encoder), 2))]
    else:
        enc_weights = [ae.encoder[i].weight_r.T for i in reversed(range(0, len(ae.encoder), 2))]

    term2 = lambd * torch.norm(torch.chain_matmul(*enc_weights))
    contr_loss = torch.mean(term1 + term2, 0)
    if reduction == 'mean':
        return torch.mean(contr_loss)
    elif reduction == 'sum':
        return torch.sum(contr_loss)
    else:
        raise ValueError(f"value for 'reduction' must be 'mean' or 'sum', got {reduction}")

# endregion