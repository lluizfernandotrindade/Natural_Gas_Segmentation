import math
import numpy as np
import tensorflow as tf 

import src.plotter


class TrainMonitor(tf.keras.callbacks.Callback):
    def __init__(self, data, epoch_interval=None):
        self.epoch_interval = epoch_interval
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            
            src.plotter.plot_result(next(iter(self.data)),
                                self.model,
                                self.model.mask_ratio
                               )
            
            
            
def cosine_schedule(base_lr, total_steps, warmup_steps):
    def step_fn(epoch):
        lr = base_lr
        epoch += 1

        progress = (epoch - warmup_steps) / float(total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        
        lr = lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        if warmup_steps:
            lr = lr * tf.minimum(1.0, epoch / warmup_steps)

        return lr

    return step_fn