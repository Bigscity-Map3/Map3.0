import importlib
import numpy as np
from torch.utils.data import DataLoader
import copy

from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch, BatchPAD
# getattr(importlib.import_module('libcity.data.dataset.dataset_subclass'),
#                            'Node2VecDataset')({})
getattr(importlib.import_module('libcity.data.dataset.dataset_subclass'),
                           'ZEMobDataset')({})