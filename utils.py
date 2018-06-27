import os
import numpy as np
from keras.utils import Sequence


class EmbedGenerator(Sequence):
    def __init__(self, embedding, X_input, batch_size, shuffle=True):
        self.embedding = embedding
        self.X_input = X_input
        self.batch_size = batch_size
        self.shuffle = shuffle

        if isinstance(self.X_input, dict):
            self.num_data = self.X_input['num_data']
        else:
            self.num_data = len(self.X_input)
        self.indexes = np.arange(self.num_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if isinstance(self.X_input, dict):
            X = {}
            for key in self.X_input.keys():
                if key == 'num_data':
                    continue
                X[key] = self.X_input[key][indexes]
        else:
            X = self.X_input[indexes]
        y = self.embedding.predict(X)
        return X, y


class SwapGenerator(Sequence):
    def __init__(self, X_input, batch_size, swap_rate=0.15, shuffle=True):
        self.X_input = X_input
        self.batch_size = batch_size
        self.swap_rate = swap_rate
        self.shuffle = shuffle

        self.num_data = len(self.X_input)
        self.indexes_1 = np.arange(self.num_data)
        self.indexes_2 = np.arange(self.num_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes_1 = np.arange(self.num_data)
        self.indexes_2 = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indexes_1)
            np.random.shuffle(self.indexes_2)

    def __len__(self):
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        indexes_1 = self.indexes_1[index * self.batch_size:(index + 1) * self.batch_size]
        indexes_2 = self.indexes_1[index * self.batch_size:(index + 1) * self.batch_size]
        X_original = self.X_input[indexes_1]
        X_ref = self.X_input[indexes_2]
        X_noise = X_original.copy()
        num_cols = X_original.shape[1]
        num_swaps = int(num_cols * self.swap_rate)
        for i in range(X_original.shape[1]):
            swap_idx = np.random.choice(num_cols, num_swaps, replace=False)
            X_noise[i, swap_idx] = X_ref[i, swap_idx]

        return X_noise, X_original
