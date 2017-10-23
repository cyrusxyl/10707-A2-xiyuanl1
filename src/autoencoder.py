import numpy as np
from scipy.special import expit
import utilities
import IPython


class AutoEncoder(object):
    def __init__(self, num_x_units, num_h_units, W=None, bias_x_tilde=None, bias_h=None):
        self.num_x_units_ = num_x_units
        self.num_h_units_ = num_h_units

        if not W:
            # N(0, 0.01)
            self.W_ = 0.1 * np.random.randn(num_h_units, num_x_units)
        elif W.shape == (num_h_units, num_x_units):
            self.W_ = W
        else:
            raise ValueError('W(weight): dimension mismatch')

        if not bias_x_tilde:
            self.bias_x_tilde_ = np.zeros((num_x_units, 1))
        elif bias_x_tilde.shape == (num_x_units, 1):
            self.bias_x_tilde_ = bias_x_tilde
        else:
            raise ValueError('shape of bias_x_tilde should match (num_x_units, 1).')

        if not bias_h:
            self.bias_h_ = np.zeros((num_h_units, 1))
        elif bias_h.shape == (num_h_units, 1):
            self.bias_h_ = bias_h
        else:
            raise ValueError('shape of bias_h should match (num_h_units, 1).')

    def train(self,
              train_data,
              valid_data,
              is_noisy=False,
              noise_prob=0.10,
              max_epoch=100,
              tol=1e-3,
              eta=0.01):
        (num_entries, num_inputs) = train_data.shape
        if num_inputs != self.num_x_units_:
            raise ValueError('column dimension of train_data must match num_x_units.')

        train_loss_record = []
        valid_loss_record = []
        diff_loss = float('inf')
        i_epoch = 0
        while (i_epoch < max_epoch) and (diff_loss > tol):
            if is_noisy:
                noisy_train_data = self.make_noisy(train_data, noise_prob)
            else:
                noisy_train_data = train_data

            order = np.random.permutation(num_entries)
            for n in order:
                x = train_data[n, :, np.newaxis]
                noisy_x = noisy_train_data[n, :, np.newaxis]

                self.gd_update(input_x=noisy_x, desired_output_x=x, eta=eta)

            train_loss = self.cross_entropy_loss(input=train_data, desired_output=train_data)
            train_loss_record.append(train_loss)

            valid_loss = self.cross_entropy_loss(input=valid_data, desired_output=valid_data)
            valid_loss_record.append(valid_loss)

            if i_epoch >= 1:
                diff_loss = abs(train_loss_record[i_epoch - 1] - train_loss_record[i_epoch])
            i_epoch += 1
            print('epoch=%d, train_loss=%f, valid_loss=%f, diff_loss=%f') % (i_epoch, train_loss,
                                                                             valid_loss, diff_loss)

        return self.W_, train_loss_record, valid_loss_record

    def cross_entropy_loss(self, input, desired_output):
        loss = 0
        (num_entries, num_inputs) = input.shape
        if num_inputs != self.num_x_units_:
            raise ValueError('column dimension of data must match num_x_units.')

        # for n in xrange(num_entries):
        #     desired_output_x = desired_output[n, :, np.newaxis]
        #     input_x = input[n, :, np.newaxis]
        #
        #     (h, x_tilde) = self.get_x_tilde(input_x)
        (_, x_tilde) = self.get_x_tilde(np.transpose(input))
        x_tilde = np.transpose(x_tilde)

        loss -= np.sum(
            np.multiply(desired_output, np.ma.log(x_tilde)).filled(0)) / num_entries
        loss -= np.sum(
            np.multiply((1 - desired_output), np.ma.log(1 - x_tilde)).filled(0)) / num_entries

        return loss

    def make_noisy(self, train_data, noise_prob=0):
        (row, col) = train_data.shape
        dropout = np.random.rand(row, col) >= noise_prob
        noisy_train_data = np.logical_and(train_data, dropout)
        return noisy_train_data

    def get_x_tilde(self, input_x):
        h = expit(self.bias_h_ + self.W_.dot(input_x))
        x_tilde = expit(self.bias_x_tilde_ + self.W_.transpose().dot(h))

        return h, x_tilde

    def gd_update(self, input_x, desired_output_x, eta):
        (h, x_tilde) = self.get_x_tilde(input_x)
        grad_bias_x_tilde = x_tilde - desired_output_x
        # deactiv = np.multiply(x_tilde, 1-x_tilde)
        # grad_a = np.multiply(grad_bias_x_tilde, deactiv)
        grad_Wt = np.dot(grad_bias_x_tilde, np.transpose(h))
        # print grad_Wt.shape

        deactiv = np.multiply(h, 1 - h)
        grad_bias_h = np.multiply(self.W_.dot(grad_bias_x_tilde), deactiv)
        grad_W = np.dot(grad_bias_h, np.transpose(input_x))
        # print grad_W.shape

        self.W_ -= eta * (grad_W + grad_Wt.transpose())
        self.bias_x_tilde_ -= eta * grad_bias_x_tilde
        self.bias_h_ -= eta * grad_bias_h


if __name__ == '__main__':
    learner = AutoEncoder(28 * 28, 100)

    # part a, part b
    (train_data_, _) = utilities.import_data('../data/digitstrain.txt')
    (valid_data_, _) = utilities.import_data('../data/digitsvalid.txt')

    (W, train_loss, valid_loss) = learner.train(train_data_, valid_data_, max_epoch=50)

    utilities.plot_weights(W, 100, 28*28)
    utilities.plot_loss(train_loss, valid_loss)
    utilities.save_weights('../results/ae_w.txt', W)
