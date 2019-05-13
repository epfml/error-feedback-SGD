"""
Makes the hyper-parameters selection easier for each experiment.
"""


def get_experiment_hyperparameters(model, dataset, optimizer):
    """
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :return: A dictionary with the hyper-parameters
    """
    hyperparameters = dict()
    if model != 'vgg' and model != 'vggnonorm' and model != 'resnet' and model != 'lstm':
        raise ValueError('Invalid value for model : {}'.format(model))
    if dataset != 'cifar10' and dataset != 'cifar100':
        raise ValueError('Invalid value for dataset : {}'.format(dataset))
    momentum = 0
    comp = True
    noscale = False
    memory = False
    mback = False
    mnorm = False
    if optimizer == 'sgdm':
        momentum = 0.9
        comp = False
    elif optimizer == 'ssgd':
        noscale = True
    elif optimizer == 'sssgd':
        pass
    elif optimizer == 'ssgdf':
        memory = True
        mback = True
        mnorm = True
    elif optimizer == 'signum':
        noscale = True
        momentum = 0.9
    else:
        raise ValueError('Invalid value for optimizer : {}'.format(optimizer))

    hyperparameters['momentum'] = momentum
    hyperparameters['comp'] = comp
    hyperparameters['noscale'] = noscale
    hyperparameters['memory'] = memory
    hyperparameters['mback'] = mback
    hyperparameters['mnorm'] = mnorm
    hyperparameters['weight_decay'] = 5e-4

    return hyperparameters


def get_experiment_name(model, dataset, optimizer):
    """
    Name where the experiment's results are saved
    :param model: 'vgg', 'vggnonorm', 'resnet' or 'lstm'
    :param dataset: 'cifar10' or 'cifar100'
    :param optimizer: 'sgdm', 'ssgd' or 'sssgd'
    :return: The name of the experiment
    """
    return model + '-' + dataset + '-' + optimizer + '/'
