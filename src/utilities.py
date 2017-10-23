import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d

"""
Useful utility functions
"""

def import_data(filename):
    dataset = np.loadtxt(filename, delimiter=',')
    (num_entries, _) = dataset.shape

    data = np.rint(dataset[:, :-1])
    labels = dataset[:, -1]

    return (data, labels)


def plot_weights(weights, num_h_units, num_x_units):
    size = int(math.sqrt(num_x_units))
    hidden_units = weights.reshape(num_h_units, size, size)
    display = montage2d(hidden_units)
    plt.imshow(display, cmap='binary')
    plt.show()


def plot_loss(train_loss, valid_loss):
    epochs = len(train_loss)
    plt.plot(range(1, epochs + 1), train_loss, label='training loss')
    plt.plot(range(1, epochs + 1), valid_loss, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def save_weights(filename, weights):
    np.savetxt(filename, weights, delimiter=',')
