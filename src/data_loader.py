import numpy as np
from tensorflow.keras.utils import Sequence

class DataLoader(Sequence):
    def __init__(self, x_set, batch_size):
        #self.x, self.y = x_set, y_set
        self.x = x_set
        self.batch_size = batch_size
        self.shuffle = True

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)