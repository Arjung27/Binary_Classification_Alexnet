#######################################################
#
# train_model.py
# Train and save models

############################################################

from learning_module import train, test, adjust_learning_rate, to_log_file, now, get_model
from learning_module import get_transform, plot_fig
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision
import argparse
import numpy as np
import os
from tqdm import tqdm


def main():
    print("\n_________________________________________________\n")
    print(now(), "train_model.py main() running.")

    parser = argparse.ArgumentParser(description='Poisoning Benchmark')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', nargs='+', default=[100, 150], type=int, help='how often to decrease lr')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    parser.add_argument('--model', default='ResNet18', type=str, help='model for training')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset')
    parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
    parser.add_argument('--output', default='output_default', type=str, help='output subdirectory')
    parser.add_argument('--checkpoint', default='check_default', type=str, help='where to save the network')
    parser.add_argument('--model_path', default='', type=str, help='where is the model saved?')
    parser.add_argument('--save_net', action='store_true', help='save net?')
    parser.add_argument('--seed', default=0, type=int, help='seed for seeding random processes.')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    parser.add_argument('--train_augment', dest='train_augment', action='store_true')
    parser.add_argument('--no-train_augment', dest='train_augment', action='store_false')
    parser.set_defaults(train_augment=False)
    parser.add_argument('--test_augment', dest='test_augment', action='store_true')
    parser.add_argument('--no-test_augment', dest='test_augment', action='store_false')
    parser.set_defaults(test_augment=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_log = "train_log_{}.txt".format(args.model)
    to_log_file(args, args.output, train_log)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ####################################################
    #               Dataset

    transform_train = get_transform(args.normalize, args.train_augment, dataset="imagenet")
    transform_test = get_transform(args.normalize, args.test_augment, dataset="imagenet")
    trainset = torchvision.datasets.ImageFolder("/cmlscratch/arjgpt27/projects/ENPM673/DL/dataset/train",
                                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    testset = torchvision.datasets.ImageFolder("/cmlscratch/arjgpt27/projects/ENPM673/DL/dataset/val",
                                                transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=4, shuffle=False)


    ####################################################

    ####################################################
    #           Network and Optimizer
    net = get_model(args.model)
    to_log_file(net, args.output, train_log)
    net = net.to(device)
    start_epoch = 0

    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=2e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()

    if args.model_path != '':
        print("loading model from path: ", args.model_path)
        state_dict = torch.load(args.model_path, map_location=device)
        net.load_state_dict(state_dict['net'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
    ####################################################

    ####################################################
    #        Train and Test
    print("==> Training network...")
    loss = 0
    all_losses = []
    all_acc = []

    all_losses_test = []
    all_acc_test = []

    epoch = start_epoch
    for epoch in tqdm(range(start_epoch, args.epochs)):
        adjust_learning_rate(optimizer, epoch, args.lr_schedule, args.lr_factor)
        loss, acc = train(net, trainloader, optimizer, criterion, device)
        all_losses.append(loss)
        all_acc.append(acc)

        if args.save_net and (epoch + 1) % 10 == 0:
            state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
                    }
            out_str = os.path.join(args.checkpoint, args.model +
                                   '_seed_' + str(args.seed) +
                                   '_normalize_' + str(args.normalize) +
                                   '_augment_' + str(args.train_augment) +
                                   '_optimizer_' + str(args.optimizer) +
                                   '_epoch_' + str(epoch) + '.t7')
            print('saving model to: ', args.checkpoint, ' out_str: ', out_str)

            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            torch.save(state, out_str)

        if (epoch + 1) % args.val_period == 0:
            print("Epoch: ", epoch)
            print("Loss: ", loss)
            print("Training acc: ", acc)
            natural_acc, test_loss = test(net, testloader, device, criterion)
            all_losses_test.append(test_loss)
            all_acc_test.append(natural_acc)
            print(now(), " Natural accuracy: ", natural_acc, "Test Loss: ", test_loss)
            to_log_file({"epoch": epoch, "loss": loss, "training_acc": acc, "natural_acc": natural_acc},
                        args.output, train_log)

    # test
    # natural_acc, test_loss = test(net, testloader, device, criterion)
    # all_losses_test.append(test_loss)
    # all_acc_test.append(natural_acc)
    # print(now(), " Natural accuracy: ", natural_acc)
    to_log_file({"epoch": epoch, "loss": loss, "natural_acc": natural_acc}, args.output, train_log)
    ####################################################

    ####################################################
    #        Save
    if args.save_net:
        state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
                }
        out_str = os.path.join(args.checkpoint, args.model +
                               '_seed_' + str(args.seed) +
                               '_normalize_' + str(args.normalize) +
                               '_augment_' + str(args.train_augment) +
                               '_optimizer_' + str(args.optimizer) +
                               '_epoch_' + str(epoch) + '.t7')
        print('saving model to: ', args.checkpoint, ' out_str: ', out_str)
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save(state, out_str)
    # plot_loss(all_losses, args)
    ####################################################

    total_epochs = np.arange(start_epoch, args.epochs)
    filename = './plots/training_acc.png'
    plot_fig(total_epochs, all_acc, filename)
    filename = './plots/test_acc.png'
    plot_fig(total_epochs, all_acc_test, filename)
    filename = './plots/train_loss.png'
    plot_fig(total_epochs, all_losses, filename)
    filename = './plots/test_loss.png'
    plot_fig(total_epochs, all_losses_test, filename)

    return


if __name__ == "__main__":
    main()
