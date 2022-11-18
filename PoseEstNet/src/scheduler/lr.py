"""lr"""

def get_lr(lr, total_epochs, steps_per_epoch, lr_step, gamma):
    """get_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    lr_step = [i * steps_per_epoch for i in lr_step]

    for i in range(total_steps):
        if i < lr_step[0]:
            lr_each_step.append(lr)
        elif i < lr_step[1]:
            lr_each_step.append(lr * gamma)
        else:
            lr_each_step.append(lr * gamma * gamma)

    return lr_each_step
