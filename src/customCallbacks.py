import src.plotter
import numpy as np
import tensorflow as tf 


class TrainMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_data, epoch_interval=None):
        self.epoch_interval = epoch_interval
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            test_images = next(iter(self.test_data))
            
            plotter.plot_result(test_images,
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