"""
Script used to explore the behavior of the ratio of the L1 norm by the L2 norm of
the memory of the optimizer.
"""

from main import construct_and_train
from utils.hyperparameters import get_experiment_hyperparameters, get_experiment_name
from tune_lr import get_tuned_learning_rate

base_folder = 'norm_ratio_experiments/'


def run_experiment(model, dataset, optimizer, prefix='', batch_size=128):
    base_name = base_folder + 'batchsize-' + str(batch_size) + '/' \
                + prefix + get_experiment_name(model, dataset, optimizer)

    hyperparameters = get_experiment_hyperparameters(model, dataset, optimizer)
    momentum = hyperparameters['momentum']
    weight_decay = hyperparameters['weight_decay']
    comp = hyperparameters['comp']
    noscale = hyperparameters['noscale']
    memory = hyperparameters['memory']
    mnorm = hyperparameters['mnorm']
    mback = hyperparameters['mback']
    norm_ratio = True

    num_epochs = [100, 50, 50]

    resume = False
    name = base_name + '/'
    lr = get_tuned_learning_rate(model, dataset, optimizer) * batch_size / 128
    print('Tuned lr : {}'.format(lr))
    for epochs in num_epochs:
        construct_and_train(name=name, dataset=dataset, model=model, resume=resume, epochs=epochs,
                            lr=lr, batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
                            comp=comp, noscale=noscale, memory=memory, mnorm=mnorm, mback=mback, norm_ratio=norm_ratio)
        resume = True
        lr /= 10


if __name__ == '__main__':
    run_experiment('vgg', 'cifar10', 'ssgdf', batch_size=32, prefix='2')
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=32, prefix='3')
