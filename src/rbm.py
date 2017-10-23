import numpy as np
from scipy.special import expit
import utilities


class RBM(object):
    def __init__(self,
                 num_x_units,
                 num_h_units,
                 cd_steps=1,
                 eval_steps=1,
                 W=None,
                 bias_x=None,
                 bias_h=None):
        """
        initialize RBM learner
        :param num_x_units: number of input units
        :param num_h_units: number of hidden units
        :param cd_steps: steps of CD
        :param eval_steps: steps of CD for computing loss
        :param W: weights
        :param bias_x: input bias
        :param bias_h: hidden bias
        """
        self.num_x_units_ = num_x_units
        self.num_h_units_ = num_h_units
        self.cd_steps_ = cd_steps

        # weight
        if not W:
            # N(0, 0.01)
            self.W_ = 0.1 * np.random.randn(num_h_units, num_x_units)
        elif W.shape == (num_h_units, num_x_units):
            self.W_ = W
        else:
            raise ValueError('W(weight): dimension mismatch')

        # bias terms
        if not bias_x:
            self.bias_x_ = np.zeros((num_x_units, 1))
        elif bias_x.shape == (num_x_units, 1):
            self.bias_x_ = bias_x
        else:
            raise ValueError('c(input bias): dimension mismatch')

        if not bias_h:
            self.bias_h_ = np.zeros((num_h_units, 1))
        elif bias_h.shape == (num_h_units, 1):
            self.bias_h_ = bias_h
        else:
            raise ValueError('b(input bias): dimension mismatch')

    def train(self, train_data, valid_data, max_epoch=100, tol=1e-3, eta=0.01):
        """
        training RBM learner
        :param train_data: training data
        :param valid_data: validation data
        :param max_epoch: maximum number of epoch
        :param tol: minimum change in loss
        :param eta: learning rate
        :return: (learnt weight, training loss recording, validation loss recording)
        """
        (num_entries, num_inputs) = train_data.shape
        if num_inputs != self.num_x_units_:
            raise ValueError('column dimension of train_data must match num_x_units.')

        # training
        train_loss_record = []
        valid_loss_record = []
        diff_loss = float('inf')
        i_epoch = 0

        while (i_epoch < max_epoch) and (diff_loss > tol):
            # scan through training sample using random order
            order = np.random.permutation(num_entries)
            for t in order:
                # update gradient using one sample
                x_t = train_data[t, :, np.newaxis]
                self.gd_update(x_t, eta)

            # track cross entropy loss
            train_loss = self.get_loss(train_data)
            valid_loss = self.get_loss(valid_data)
            train_loss_record.append(train_loss)
            valid_loss_record.append(valid_loss)

            # stopping criteria
            if i_epoch >= 1:
                diff_loss = abs(train_loss_record[i_epoch - 1] - train_loss_record[i_epoch])
            i_epoch += 1
            print('epoch=%d, train_loss=%f, valid_loss=%f, diff_loss=%f') % (i_epoch, train_loss,
                                                                             valid_loss, diff_loss)

            # utilities.plot_weights(self.W_, self.num_h_units_, self.num_x_units_)
            # utilities.plot_loss(train_loss_record, valid_loss_record)

        return self.W_, train_loss_record, valid_loss_record

    def get_loss(self, data):
        """
        compute loss of RBM on given data
        :param data: input data
        :return: loss
        """
        loss = 0
        (num_entries, num_inputs) = data.shape
        if num_inputs != self.num_x_units_:
            raise ValueError('column dimension of data must match num_x_units.')

        (_, x_tilde) = self.get_x_tilde(np.transpose(data))
        x_tilde = np.transpose(x_tilde)

        # loss = - (x log(p) + (1-x) log (1-p))
        loss -= np.sum(np.multiply(data, np.log(x_tilde))) / num_entries
        loss -= np.sum(np.multiply((1 - data), np.log(1 - x_tilde))) / num_entries

        return loss

    def gd_update(self, x, eta):
        """
        gradient descent on parameters
        :param x: current input
        :param eta: learning rate
        """
        # positive sample
        grad_pos = self.get_energy_grad(x)

        # negative sample from Gibbs
        (neg_h, neg_x) = self.gibbs_chain(init_x=x, cd_steps=self.cd_steps_)
        grad_neg = self.get_energy_grad(neg_x)

        # gradient update
        self.W_ += eta * (grad_pos[0] - grad_neg[0])
        self.bias_x_ += eta * (grad_pos[1] - grad_neg[1])
        self.bias_h_ += eta * (grad_pos[2] - grad_neg[2])

    def get_energy_grad(self, x):
        """
        compute the energy gradient given current input
        :param x: current input
        :return: gradient in parameters
        """
        h_mean = expit(self.bias_h_ + self.W_.dot(x))
        grad_W = h_mean.dot(x.transpose())
        grad_bias_x = x
        grad_bias_h = h_mean

        return grad_W, grad_bias_x, grad_bias_h

    def gibbs_chain(self, init_x=None, cd_steps=1):
        """
        :param init_x: initial point of sampling
        :param cd_steps: number of steps for CD
        :return: sampled units
        """
        # initialization
        if init_x is None:
            x = np.random.binomial(n=1, p=0.5, size=(self.num_x_units_, 1))
        else:
            x = init_x

        # gibbs sampling with step cd_steps
        for k in xrange(cd_steps):
            h = self.gibbs_sample_h(x)
            x = self.gibbs_sample_x(h)

        return h, x

    def gibbs_sample_h(self, x):
        h_mean = expit(self.bias_h_ + self.W_.dot(x))
        return self.bernoulli(h_mean)

    def gibbs_sample_x(self, h):
        x_tilde = expit(self.bias_x_ + self.W_.transpose().dot(h))
        return self.bernoulli(x_tilde)

    def bernoulli(self, means):
        (row, col) = means.shape
        uniform = np.random.rand(row, col)
        sample = means >= uniform
        return sample

    def get_x_tilde(self, x_t, eval_steps=1):
        """
        making prediction with current parameters
        :param x_t: current input
        :return: prediction
        """
        # obtain hidden layer from gibbs sampling
        x = x_t
        for k in xrange(eval_steps):
            h = self.gibbs_sample_h(x)
            x = self.gibbs_sample_x(h)

        # sigmoid
        x_tilde = expit(self.bias_x_ + self.W_.transpose().dot(h))

        return h, x_tilde


if __name__ == '__main__':
    learner = RBM(28 * 28, 500)

    # part a, part b
    (train_data_, _) = utilities.import_data('../data/digitstrain.txt')
    (valid_data_, _) = utilities.import_data('../data/digitsvalid.txt')

    (W, train_loss, valid_loss) = learner.train(train_data_, valid_data_, max_epoch=30, eta=0.01)
    print W.shape

    # utilities.plot_weights(W, 50, 28*28)
    utilities.plot_loss(train_loss, valid_loss)
    # utilities.save_weights('../results/rbm_w.txt', W)
    #
    # # part c
    # samples = []
    # random_config = np.random.rand(100, 784)
    # (_, samples) = learner.get_x_tilde(random_config.transpose(), eval_steps=1000)
    #
    # print samples.shape
    # utilities.plot_weights(samples.transpose(), 100, 28*28)
