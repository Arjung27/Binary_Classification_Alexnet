############################################################
#
# learning_module.py
# Python module for deep learing
#
############################################################

import datetime
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from matplotlib.ticker import FormatStrFormatter
# from models import *
import torch
import torch.nn as nn
from ImageNet_Models.alexnet_fc7out import AlexNet
import torchvision.transforms as transforms
import torchvision
import csv

data_mean = [0.4886, 0.4549, 0.4178]
data_std = [0.2606, 0.2552, 0.2580]

def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, 'a') as f:
        f.write(str(now()) + " " + str(out_dict) + "\n")

    print('logging done in ' + out_dir + '.')


def to_results_table(stats, out_dir, log_name="results.csv"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)
    try:
        with open(fname, 'r') as f:
            pass
    except:
        with open(fname, 'w') as f:
            fieldnames = list(stats.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(fname, 'a') as f:
        fieldnames = list(stats.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(stats)

    print('results logged in  ' + out_dir + ',' + ' in ' + log_name)


def adjust_learning_rate(optimizer, epoch, lr_schedule, lr_factor):
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_factor
        print("Adjusting learning rate ", param_group['lr'] / lr_factor, "->", param_group['lr'])
    return

def test(net, testloader, device, criterion):

    net.eval()
    test_loss = 0
    natural_correct = 0
    total = 0
    results = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)
            natural_outputs = net(inputs)
            loss = criterion(natural_outputs, targets)
            test_loss += loss.item()
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)

    test_loss = test_loss / (batch_idx + 1)
    natural_acc = 100. * natural_correct / total

    results['Clean acc'] = natural_acc

    return natural_acc, test_loss


def train(net, trainloader, optimizer, criterion, device):
    """ Function to perform one epoch of training
    input:
        net: pytorch network object
        trainloader: pytorch dataloader object
        optimizer: pytorch optimizer object
        criterion: loss function

    output:
        train_loss: float, average loss value
        acc: float, percetage of training data correctly labeled
    """

    # Set net to train and zeros stats
    net.train()
    net = net.to(device)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            print('{}/{} training accuracy: {}'.format(batch_idx, len(trainloader), correct*100/total))
    
    train_loss = train_loss / (batch_idx + 1)
    acc = 100. * correct / total
    print('train loss for epoch: ', train_loss)
    print('train accuracy for epoch: ', acc)

    return train_loss, acc


def get_transform(normalize, augment, dataset="CIFAR10"):

    mean = data_mean
    std = data_std
    cropsize = 224
    padding = None

    transform_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    if normalize and augment:
        transform_list.extend([
            transforms.RandomCrop(cropsize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    elif augment:
        transform_list.extend([
            transforms.RandomCrop(cropsize, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif normalize:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_model(model):

    if model.upper() == 'ALEXNET':
        net = AlexNet(feature_size=4096)
    elif model == 'Resnet':
        net = ResNet18()
    
    return net


def load_model_from_checkpoint(model, model_path, dataset):
    net = get_model(model, dataset)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict['net'])

    except:
        net = torch.nn.DataParallel(net).cuda()
        net.eval()
        print('==> Resuming from checkpoint for %s..' % model)
        checkpoint = torch.load(model_path)
        if 'module' not in list(checkpoint['net'].keys())[0]:
            # to be compatible with DataParallel
            net.module.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint['net'])

    return net

def plot_fig(x_data, y_data, path):

    fig = plt.figure()
    plt.plot(x_data, y_data)
    plt.savefig(path)
