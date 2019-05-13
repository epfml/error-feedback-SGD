"""
Main experiments that were conducted, after the learning rate tuning.
"""

from main import construct_and_train
from utils.hyperparameters import get_experiment_hyperparameters, get_experiment_name
from tune_lr import get_tuned_learning_rate


base_folder = 'main_experiments/'


def run_experiment(model, dataset, optimizer, prefix='', batch_size=128, num_exp=3, start_at=1):
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

    num_epochs = [100, 50, 50]

    for exp_index in range(start_at, num_exp + start_at):
        resume = False
        name = base_name + str(exp_index) + '/'
        lr = get_tuned_learning_rate(model, dataset, optimizer)*batch_size/128
        print('Tuned lr : {}'.format(lr))
        for epochs in num_epochs:
            construct_and_train(name=name, dataset=dataset, model=model, resume=resume, epochs=epochs,
                                lr=lr, batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
                                comp=comp, noscale=noscale, memory=memory, mnorm=mnorm, mback=mback)
            resume = True
            lr /= 10


if __name__ == '__main__':
    # run_experiment('vgg', 'cifar10', 'sgdm', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'ssgdf', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'signum', batch_size=8)
    # run_experiment('vgg', 'cifar10', 'sssgd', batch_size=8)

    # run_experiment('vggnonorm', 'cifar10', 'sgdm', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'ssgdf', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'signum', batch_size=128)
    # run_experiment('vggnonorm', 'cifar10', 'sssgd', batch_size=128)
    
    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'signum', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=128)

    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'signum', batch_size=8)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=8)

    # run_experiment('resnet', 'cifar100', 'sgdm', batch_size=32)
    # run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=32)
    run_experiment('resnet', 'cifar100', 'ssgdf', batch_size=128)
    # run_experiment('resnet', 'cifar100', 'sssgd', batch_size=32)
