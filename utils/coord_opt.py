import numpy as np


class MSE(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(weights):
        """
        :param weights: matrix of shape (dimension, 1)
        :return: scalar
        """
        return 0.5*np.linalg.norm(weights)**2

    @staticmethod
    def compute_grad(weights):
        stoch_grad = weights.copy()
        stoch_grad[0] += np.random.normal(0, 100)        
        return stoch_grad


def scaled_sign(x):
    return np.linalg.norm(x, ord=1)/len(x)*np.sign(x)


def unscaled_sign(x):
        return np.sign(x)


class SGD(object):
    def __init__(self, loss, sign=False, scale= True, comp=None, memory=None, lr=0.001, momentum=0):
        self.loss = loss
        self.comp = comp
        if sign and scale:
            self.comp = scaled_sign
        if sign and not scale:
            self.comp = unscaled_sign
        if memory is not None and memory:
            self.memory = 0
        else:
            self.memory = None
        self.momentum = momentum
        if momentum:
            self.momentum_buffer = 0
        self.lr = lr

    def step(self, weights):
        d_p = self.loss.compute_grad(weights)
        if self.momentum:
            self.momentum_buffer += self.momentum*d_p
            d_p = self.momentum_buffer
        d_p = self.lr*d_p
        if self.memory is not None:
            g = d_p + self.memory
        else:
            g = d_p
        if self.comp is not None:
            g = self.comp(g)
        ''' hack to scale the signed gradient by learning rate since
            torch.sign(x) ignores learning rate '''
        if self.comp == unscaled_sign:
            g = self.lr*g
        if self.memory is not None:
            self.memory += d_p - g
        return g


def initialize_parameters(n_params):
    return np.random.normal(0,1,n_params)[:,None]


def train_model(optimizer=None, loss=MSE(), epochs=100, lr=0.0001, repeats=100):
    res = dict()
    if optimizer is None:
        optimizer = SGD(loss, lr=lr)
    train_losses_repeats = []
    for i in range(repeats):
        w = initialize_parameters(100)
        train_losses = [loss.compute_loss(w)]
        for epoch in range(epochs):
            train_loss = 0
            nb_corrects = 0
            gradient = optimizer.step(w)
            w = w - gradient
            train_loss = loss.compute_loss(w)
            train_losses.append(train_loss)
        train_losses_repeats.append(train_losses)
    res['losses'] = np.array(train_losses_repeats)
    return res
