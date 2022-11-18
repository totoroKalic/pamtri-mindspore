"""optimizers"""
import mindspore.nn as nn

def init_optim(optim, params, lr, weight_decay):
    """choose optimizer"""
    if optim == 'adam':
        return nn.Adam(params, learning_rate=lr, weight_decay=weight_decay, loss_scale=64)
    if optim == 'amsgrad':
        return nn.Adam(params, learning_rate=lr, weight_decay=weight_decay, use_nesterov=True, loss_scale=64)
    if optim == 'sgd':
        return nn.SGD(params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay, loss_scale=64)
    if optim == 'rmsprop':
        return nn.RMSProp(params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay, loss_scale=64)

    raise KeyError("Unsupported optimizer: {}".format(optim))
