import numpy as np


class MSE(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(predictions, targets):
        """
        :param predictions: predictions matrix of shape (num_samples, 1)
        :param targets: targets matrix of shape (num_samples, 1)
        :return: matrix of shape (num_samples, 1) containing the squared errors (MSE is the mean of this tensor).
        """
        return ((targets-predictions)**2).sum()

    @staticmethod
    def compute_grad(batch_x, batch_y, predictions):
        return batch_x.T@(predictions - batch_y)/batch_x.shape[0]


def scaled_sign(x):
    return np.linalg.norm(x, ord=1)/len(x)*np.sign(x)


def unscaled_sign(x):
        return np.sign(x)


class TemporarilyAddMemory:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __enter__(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                memory = param_state['memory']
                p.data.add_(1, memory)

    def __exit__(self, type, value, traceback):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                memory = param_state['memory']
                p.data.add_(-1, memory)


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

    def step(self, batch_x, batch_y, preds):
        d_p = self.loss.compute_grad(batch_x, batch_y, preds)
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


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size, :], y[i:i + batch_size])


def initialize_parameters(n_params):
    return np.zeros((n_params, 1))


def train_model(x_train, y_train, x_test=None, y_test=None, optimizer=None, loss=MSE(), batch_size=16, epochs=100, lr=0.01, fully_corrective = False):
    res = dict()
    if optimizer is None:
        optimizer = SGD(loss, lr=lr)
    assert (optimizer.memory is not None or not fully_corrective), "if using fully_corrective, optimizer.memory cannot be None."
    w_size = x_train.shape[1]
    w = initialize_parameters(w_size)
    train_losses = []
    test_losses = []
    w_list = [w]
    gradients_list = []
    if batch_size <= 0:
        batch_size = x_train.shape[0]
    for epoch in range(epochs):
        train_loss = 0
        nb_corrects = 0
        for (batch_x, batch_y) in next_batch(x_train, y_train, batch_size):
            preds = batch_x@w
            gradient = optimizer.step(batch_x, batch_y, preds)
            gradients_list.append(gradient)
            w = w - gradient
            if fully_corrective:
                w_list.append(w + optimizer.memory)
            else:
                w_list.append(w)
            if fully_corrective:
                corrected_preds = batch_x@(w + optimizer.memory)
            else:
                corrected_preds = preds
            train_loss += loss.compute_loss(corrected_preds, batch_y)
        train_losses.append(train_loss/x_train.shape[0])
        if x_test is not None and y_test is not None:
            if fully_corrective:
                corrected_preds = x_test@(w - optimizer.memory)
            else:
                corrected_preds = x_test@w
            test_losses.append(loss.compute_loss(corrected_preds, y_test)/x_test.shape[0])
    res['train_losses'] = np.array(train_losses)
    res['test_losses'] = np.array(test_losses)
    res['w'] = np.array(w_list)
    res['gradients'] = np.array(gradients_list)
    return res
