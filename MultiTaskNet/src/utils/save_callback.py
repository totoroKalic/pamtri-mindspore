"""saveCallBack"""
import os
import time
from mindspore import load_param_into_net, load_checkpoint
from mindspore.train.callback import Callback
from src.utils.evaluate import test

class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """
    def __init__(self, model, query_dataset, gallery_dataset,
                 vcolor2label, vtype2label, epoch_per_eval, max_epoch, path, step_size):
        super(SaveCallback, self).__init__()
        self.model = model
        self.query_dataset = query_dataset
        self.gallery_dataset = gallery_dataset
        self.vcolor2label = vcolor2label
        self.vtype2label = vtype2label
        self.epoch_per_eval = epoch_per_eval
        self.max_epoch = max_epoch
        self.path = path
        self.step_size = step_size

    def epoch_end(self, run_context):
        """
        eval and save model while training.
        """
        t1 = time.time()
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        file_name = self.path + "MultipleNet-" + str(cur_epoch) + "_" + str(self.step_size) + ".ckpt"
        param_dict = load_checkpoint(file_name)
        load_param_into_net(self.model, param_dict)

        print("\n--------------------{} / {}--------------------\n".format(cur_epoch, self.max_epoch))
        print("----------device is {}".format(int(os.getenv('DEVICE_ID'))))
        _ = test(self.model, True, True, self.query_dataset, self.gallery_dataset, \
            self.vcolor2label, self.vtype2label, return_distmat=True)
        t2 = time.time()
        print("Eval in training Time consume: ", t2 - t1, "\n")
