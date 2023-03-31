import json
import os
from logging import getLogger
import random
import numpy as np
from tqdm import tqdm

from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
#[2020-IJCAI Multi-View Joint Graph Representation Learning for Urban Region Embedding]
class MVURE(AbstractTraditionModel):
    def __int__(self,config,data_feature):
        return