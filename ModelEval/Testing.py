from typing import Tuple, List, Union, Generic
import numpy as np
import tensorflow as tf
import src_3D.View2D as View2D
from conf_3D.config import Configuration


class Testing(object):

    def __init__(self,
                 test_data: Tuple[np.ndarray, np.ndarray],
                 model_path: str,
                 #save_dir: str,
                 cfg: Configuration):
        self.test_data = test_data
        self.model_path = model_path
        #self.save_dir = save_dir
        self.cfg = cfg

    def inference(self):
        return

    def metrics_estimation(self, gt: np.ndarray, pred: np.ndarray):
        return

    def save_png(self):
        return

    def save_inference_sgy(self):
        return
