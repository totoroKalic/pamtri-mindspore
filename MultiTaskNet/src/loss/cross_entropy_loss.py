
"""CrossEntropyLabelSmooth"""
import mindspore
import mindspore.nn as nn

class CrossEntropyLabelSmooth(nn.Cell):
    """
    Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.zeros = mindspore.ops.Zeros()
        self.onehot = nn.OneHot(axis=1, depth=num_classes)
        self.unsqueeze = mindspore.ops.ExpandDims()
        self.logsoftmax = nn.LogSoftmax(axis=1)
        self.sum = mindspore.ops.ReduceSum(keep_dims=False)

    def construct(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = self.onehot(targets)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (- targets * log_probs).mean(axis=0, keep_dims=False)
        loss = self.sum(loss, 0)

        return loss
