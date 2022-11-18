"""get lr"""

def get_lr(lr, total_epochs, steps_per_epoch, lr_step, gamma):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs

    for _ in range(total_steps):
        lr_each_step.append(lr)

    return lr_each_step
