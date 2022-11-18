"""dataset"""
import os
import copy
import json
from pathlib import Path
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose

from .veri import VeRiDataset

def create_dataset(cfg, data_dir, is_train=True):
    """create_dataset"""
    device_num, rank_id = _get_rank_info()

    data = VeRiDataset(cfg, data_dir, is_train)

    if is_train:
        if device_num == 1:
            dataset = ds.GeneratorDataset(data, column_names=["input", "target", "target_weight"], \
                num_parallel_workers=1, shuffle=cfg.TRAIN.SHUFFLE, num_shards=1, shard_id=0)
        else:
            dataset = ds.GeneratorDataset(data, column_names=["input", "target", "target_weight"], \
                num_parallel_workers=1, shuffle=cfg.TRAIN.SHUFFLE, num_shards=device_num, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(data, \
            column_names=["input", "target", "target_weight", "center", "scale", "score", "image", \
                "joints", "joints_vis"], num_parallel_workers=1, shuffle=False, num_shards=1, shard_id=0)

    trans = Compose([
        py_vision.ToTensor(),
        py_vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = dataset.map(operations=trans, input_columns="input", num_parallel_workers=8)
    if is_train:
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)
    else:
        dataset = dataset.batch(cfg.TEST.BATCH_SIZE, drop_remainder=True, num_parallel_workers=8)

    if is_train:
        return dataset

    return data, dataset

def get_label(cfg, data_dir):
    """
    get label
    """
    lable_path = os.path.join(data_dir, 'annot/image_test.json')

    if not os.path.isfile(lable_path):
        os.mknod(lable_path)
        data = VeRiDataset(cfg, data_dir, False).db

        label = {}
        for i in range(data.__len__()):
            out = copy.deepcopy(data[i])
            label['{}'.format(i)] = out['image']

        label_json_path = Path(lable_path)
        with label_json_path.open('w') as dst_file:
            json.dump(label, dst_file)

    return lable_path

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE"))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
