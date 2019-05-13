"""
Script containing the main functions to train and evaluate the models
with several optimizers. The results can be easily saved.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import VGG, ResNet18, PreActResNet18, GoogLeNet, DenseNet121, ResNeXt29_2x64d, MobileNet, MobileNetV2, \
    DPN92, ShuffleNetG2, SENet18, ShuffleNetV2
from optimizers.ErrorFeedbackSGD import ErrorFeedbackSGD
from optimizers.TemporarilyAddMemory import TemporarilyAddMemory
from utils.progress_bar import progress_bar
from utils.pickle import save_obj, load_obj, make_directory, make_file_directory


def load_data(dataset='cifar10', batch_size=128):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader, num_classes


def build_model(device, model_name, num_classes=10):
    """
    :param device: 'cuda' if you have a GPU, 'cpu' otherwise
    :param model_name: One of the models available in the folder 'models'
    :param num_classes: 10 or 100 depending on the chosen dataset
    :return: The model architecture
    """
    print('==> Building model..')
    model_name = model_name.lower()
    if model_name == 'vgg':
        net = VGG('VGG19', num_classes=num_classes)
    elif model_name == 'vggnonorm':
        net = VGG('VGG19', num_classes=num_classes, batch_norm=False)
    elif model_name == 'resnet':
        net = ResNet18(num_classes=num_classes)
    elif model_name == 'preactresnet':
        net = PreActResNet18()
    elif model_name == 'googlenet':
        net = GoogLeNet()
    elif model_name == 'densenet':
        net = DenseNet121()
    elif model_name == 'resnext':
        net = ResNeXt29_2x64d()
    elif model_name == 'mobilenet':
        net = MobileNet()
    elif model_name == 'mobilenetv2':
        net = MobileNetV2()
    elif model_name == 'dpn':
        net = DPN92()
    elif model_name == 'shufflenetg2':
        net = ShuffleNetG2()
    elif model_name == 'senet':
        net = SENet18()
    elif model_name == 'shufflenetv2':
        net = ShuffleNetV2(1)
    else:
        raise ValueError('Error: the specified model is incorrect ({})'.format(model_name))

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net


def load_checkpoint(net, name):
    """
    Load saved weights to a given net.
    """
    print('==> Resuming from checkpoint..')
    if not os.path.isdir('checkpoints'):
        raise Exception('Error: no checkpoint directory found!')
    checkpoint = torch.load('./checkpoints/' + name + '.t7')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['acc']
    return start_epoch, best_acc


def create_optimizer(net, comp, memory, noscale, lr=0.1, momentum=0.9, weight_decay=5e-4):
    """
    Creates the right optimizer regarding to the parameters and attach it to the net's parameters.
    :param net: The net to optimize.
    :param comp: Bool (True = scaled sign compression, False = no comp)
    :param memory: If there is a compression whether to have a feedback loop or not
    :param noscale: If True the compression operator is unscaled sign (if the compression is existing)
    :param lr: Initial learning rate of the optimizer.
    :param momentum: Momentum of the optimizer.
    :param weight_decay: Weight decay of the optimizer.
    :return: A Pytorch optimizer.
    """
    if memory and not comp:
        raise ValueError('The memory option is activated without the compression operator')
    if comp:
        comp = 'scaled_sign'
        if noscale:
            comp = 'sign'
        optimizer = ErrorFeedbackSGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                     comp=comp, memory=memory)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def train(net, trainloader, device, optimizer, criterion, memory_back=False):
    """
    One epoch training of a network.
    :param net: The given network.
    :param trainloader: Pytorch DataLoader (train set)
    :param device: Either 'cuda' or 'cpu'
    :param optimizer: The used optimizer.
    :param criterion: The loss function.
    :param memory_back: Whether or not for each batch the memory of the optimizer (if it has one) should be
    temporarily added back to the net's parameters to compute the several metrics with these new parameters.
    It doesn't change the final net's parameters.
    :return: (train_loss, train_acc, train_loss_with_memory_back, train_acc_with_memory_back, L1/L2 norm ratio
              of the gradients, L1/L2 norm ratio of g)
    """
    net.train()
    train_loss = 0
    mback_train_loss = 0
    correct = 0
    mback_correct = 0
    total = 0
    norm_ratio_val = 0
    corrected_norm_ratio_val = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if memory_back:
            with TemporarilyAddMemory(optimizer):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                mback_train_loss += loss.item()
                _, predicted = outputs.max(1)
                mback_correct += predicted.eq(targets).sum().item()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        norm_ratio_val += optimizer.gradient_norms_ratio()
        corrected_norm_ratio_val += optimizer.corrected_gradient_norms_ratio()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    loss = train_loss/(batch_idx + 1)
    acc = 100. * correct/total

    return loss, acc, mback_train_loss/(batch_idx + 1), 100. * mback_correct/total, norm_ratio_val/(batch_idx + 1), \
           corrected_norm_ratio_val/(batch_idx + 1)


def test(net, testloader, device, optimizer, criterion, memory_back=False):
    """
    One test evaluation of a network.
    :param net: The given network.
    :param testloader: Pytorch DataLoader (train set)
    :param device: Either 'cuda' or 'cpu'
    :param optimizer: The used optimizer.
    :param criterion: The loss function.
    :param memory_back: Whether or not for each batch the memory of the optimizer (if it has one) should be
    temporarily added back to the net's parameters to compute the several metrics with these new parameters.
    It doesn't change the final net's parameters.
    :return: (train_loss, train_acc, train_loss_with_memory_back, train_acc_with_memory_back)
    """
    net.eval()
    test_loss = 0
    mback_test_loss = 0
    correct = 0
    mback_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if memory_back:
                with TemporarilyAddMemory(optimizer):
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    mback_test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    mback_correct += predicted.eq(targets).sum().item()

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    loss = test_loss / (batch_idx + 1)
    acc = 100. * correct / total

    return loss, acc, mback_test_loss / (batch_idx + 1), 100. * mback_correct / total


def write_results(args, res):
    """
    Write recorded training metrics to files.
    :param args: Training args.
    :param res: Results of the training.
    """
    name = args['name']
    directory = './results/' + name
    print('Writing results ({})..'.format(name))
    make_directory(directory)
    save_obj(res['train_losses'], directory + '/train_losses')
    save_obj(res['train_accuracies'], directory + '/train_accuracies')
    save_obj(res['test_losses'], directory + '/test_losses')
    save_obj(res['test_accuracies'], directory + '/test_accuracies')
    if args['mback']:
        save_obj(res['memory_back_train_losses'], directory + '/memory_back_train_losses')
        save_obj(res['memory_back_train_accuracies'], directory + '/memory_back_train_accuracies')
        save_obj(res['memory_back_test_losses'], directory + '/memory_back_test_losses')
        save_obj(res['memory_back_test_accuracies'], directory + '/memory_back_test_accuracies')
    if args['mnorm']:
        save_obj(res['memory_norms'], directory + '/memory_norms')
    if args['norm_ratio']:
        save_obj(res['gradient_norm_ratios'], directory + '/gradient_norm_ratios')
        save_obj(res['corrected_norm_ratios'], directory + '/corrected_norm_ratios')
    open_mode = 'w'
    if args['resume']:
        open_mode = 'a'
    with open(directory + '/README.md', open_mode) as file:
        if args['resume']:
            file.write('\n')
        for arg, val in args.items():
            file.write(str(arg) + ': ' + str(val) + '\\\n')


def construct_and_train(name='last_model', dataset='cifar10', model='vgg', resume=False, epochs=100,
                        lr=0.1, batch_size=128, momentum=0.9, weight_decay=5e-4,
                        comp=False, noscale=False, memory=False, mnorm=False, mback=False, norm_ratio=False):
    """
    Constructs a network, trains it, and optionally saves the results.
    :param name: Model name (using for saving)
    :param dataset: Either 'cifar10' or 'cifar100'
    :param model: Either 'vgg' or 'resnet'
    :param resume: Reload and resume a model whose training was stopped
    :param epochs: Number of epochs to train
    :param lr: Initial learning rate
    :param batch_size: Batch size
    :param momentum: Momentum of the optimizer
    :param weight_decay: Weight decay of the optimizer
    :param comp: Compression operator
    :param noscale: Doesn't scale the compression
    :param memory: Whether or not to add a memory (feedback loop) to the optimizer
    :param mnorm: Whether or not to save the memory norm
    :param mback: Whether or not to save metrics when the memory is added back to the net params
    :param norm_ratio: Whether or not to save norms ratios
    """
    args = dict(name=name, dataset=dataset, model=model, resume=resume, epochs=epochs,
                lr=lr, batch_size=batch_size, momentum=momentum, weight_decay=weight_decay,
                comp=comp, noscale=noscale, memory=memory, mnorm=mnorm, mback=mback, norm_ratio=norm_ratio)
    trainloader, testloader, num_classes = load_data(dataset, batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(device, model, num_classes)
    start_epoch = 0
    best_acc = 0
    if resume:
        start_epoch, best_acc = load_checkpoint(net, name)

    optimizer = create_optimizer(net, comp, memory, noscale, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    res = dict(train_losses=[],
               train_accuracies=[],
               test_losses=[],
               test_accuracies=[],
               memory_norms=[],
               memory_back_train_losses=[],
               memory_back_train_accuracies=[],
               memory_back_test_losses=[],
               memory_back_test_accuracies=[],
               gradient_norm_ratios=[],
               corrected_norm_ratios=[])
    if resume:
        path = './results/' + name
        res['train_losses'] = load_obj(path + '/train_losses')
        res['train_accuracies'] = load_obj(path + '/train_accuracies')
        res['test_losses'] = load_obj(path + '/test_losses')
        res['test_accuracies'] = load_obj(path + '/test_accuracies')
        if mback:
            res['memory_back_train_losses'] = load_obj(path + '/memory_back_train_losses')
            res['memory_back_train_accuracies'] = load_obj(path + '/memory_back_train_accuracies')
            res['memory_back_test_losses'] = load_obj(path + '/memory_back_test_losses')
            res['memory_back_test_accuracies'] = load_obj(path + '/memory_back_test_accuracies')
        if mnorm:
            res['memory_norms'] = load_obj(path + '/memory_norms')
        if norm_ratio:
            res['gradient_norm_ratios'] = load_obj(path + '/gradient_norm_ratios')
            res['corrected_norm_ratios'] = load_obj(path + '/corrected_norm_ratios')
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            print('\nEpoch: %d' % epoch)
            train_loss, train_acc, mback_train_loss,\
                mback_train_acc, norm_ratio_val, corrected_norm_ratio_val = train(net, trainloader, device, optimizer,
                                                                                  criterion, memory_back=mback)
            test_loss, test_acc, mback_test_loss, mback_test_acc = test(net, testloader, device, optimizer,
                                                                        criterion, memory_back=mback)
            res['train_losses'].append(train_loss)
            res['train_accuracies'].append(train_acc)
            res['test_losses'].append(test_loss)
            res['test_accuracies'].append(test_acc)
            if mback:
                res['memory_back_train_losses'].append(mback_train_loss)
                res['memory_back_train_accuracies'].append(mback_train_acc)
                res['memory_back_test_losses'].append(mback_test_loss)
                res['memory_back_test_accuracies'].append(mback_test_acc)
            if mnorm:
                res['memory_norms'].append(optimizer.memory_norm())
            if norm_ratio:
                res['gradient_norm_ratios'].append(norm_ratio_val)
                res['corrected_norm_ratios'].append(corrected_norm_ratio_val)
            if test_acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                make_directory('./checkpoints/' + name)
                torch.save(state, './checkpoints/' + name + '.t7')
                best_acc = test_acc
    except KeyboardInterrupt:
        print('Interrupting..')
    finally:
        write_results(args, res)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ef-signSGD experiments')
    parser.add_argument('--name', default='last_model', type=str, help='checkpoint name (default last_model)')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset (cifar 10 or 100)')
    parser.add_argument('--model', default='vgg', type=str, help='Model architecture')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD weight decay')
    parser.add_argument('--comp', action='store_true', help='apply the scaled sign compression operator')
    parser.add_argument('--noscale', action='store_true', help='apply only the sign compression operator')
    parser.add_argument('--memory', action='store_true', help='add a memory to the optimizer')
    parser.add_argument('--mnorm', action='store_true', help='computes the norm of the memory at each epoch')
    parser.add_argument('--mback', action='store_true', help='computes the train/test losses/accuracies by adding the'
                                                             'memory back')
    args = parser.parse_args()
    construct_and_train(name=args.name, dataset=args.dataset, model=args.model, resume=args.resume, epochs=args.epochs,
                        lr=args.lr, batch_size=args.bs, momentum=args.momentum, weight_decay=args.weight_decay,
                        comp=args.comp, noscale=args.noscale, memory=args.memory, mnorm=args.mnorm, mback=args.mback)
