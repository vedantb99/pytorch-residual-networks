#coding:utf-8

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *   # Keep any other data imports if needed
from models import *

import torchvision
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10
from tensorboardX import SummaryWriter

import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layer_n = int(sys.argv[1])

# Define checkpoint and log directory names based on the network depth.
ckpt_name = "checkpoints/ResNet-%d_cifar10.pth" % (layer_n*6+2)
log_name = "./logs/ResNet-%d_cifar10_log/" % (layer_n*6+2)

# Ensure the directories exist
os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
os.makedirs(log_name, exist_ok=True)

# Data path for CIFAR10 (will be used for download if not available)
data_path = '/data4/home/vedantb/datasets/cifar10'
os.makedirs(data_path, exist_ok=True)

# Use a mini-batch size of 128
batch_size = 128

def train(cnn_model, start_epoch, train_loader, test_loader, lr, auto_lr=True):
    # Terminate training at 64k iterations
    max_iter = 64000

    learning_rate = lr
    print("Initial learning rate: %f" % learning_rate)
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    train_writer = SummaryWriter(log_dir=log_name+'train')
    test_writer = SummaryWriter(log_dir=log_name+'test')

    train_iter = 0
    epoch = start_epoch
    while train_iter < max_iter:
        for batch_idx, (data_x, data_y) in enumerate(train_loader):
            # Check if we reached or exceeded max iterations
            if train_iter >= max_iter:
                break

            # Learning rate schedule:
            if auto_lr:
                if train_iter == 32000:
                    learning_rate = lr / 10.
                    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate,
                                                momentum=0.9, weight_decay=0.0001)
                    print("Learning rate decayed to %f at iteration %d" % (learning_rate, train_iter))
                if train_iter == 48000:
                    learning_rate = lr / 100.
                    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate,
                                                momentum=0.9, weight_decay=0.0001)
                    print("Learning rate decayed to %f at iteration %d" % (learning_rate, train_iter))

            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()
            output = cnn_model(data_x)
            loss = criterion(output, data_y)

            _, predicted = torch.max(output.data, 1)
            batch_size_actual = data_y.size(0)

            loss.backward()
            optimizer.step()

            if train_iter % 10 == 0:
                print("Epoch %d, Step %d/%d, iter %d, Loss: %f, lr: %f" \
                     % (epoch, batch_idx, len(train_loader), train_iter, loss.item(), learning_rate))
                train_writer.add_scalar('loss', loss.item(), train_iter)

            # Every 100 iterations, evaluate on the test set
            if train_iter % 100 == 0:
                cnn_model.eval()
                correct = 0
                total = 0
                loss_sum = 0.0
                with torch.no_grad():
                    for test_batch_idx, (images, labels) in enumerate(test_loader):
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = cnn_model(images)
                        loss_sum += criterion(outputs, labels).item()
                        _, predicted_test = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted_test == labels).sum().item()
                avg_loss = loss_sum / len(test_loader)
                test_acc = correct / total
                print("iter %d, Test Accuracy: %f, Test Loss: %f" % (train_iter, test_acc, avg_loss))
                test_writer.add_scalar('loss', avg_loss, train_iter)
                test_writer.add_scalar('accuracy', test_acc, train_iter)
                cnn_model.train()

            train_iter += 1

        epoch += 1
        # Save a checkpoint at the end of each epoch
        state_dict = {"state": cnn_model.state_dict(), "epoch": epoch, "lr": learning_rate}
        torch.save(state_dict, ckpt_name)
        print("Checkpoint saved at epoch %d: %s" % (epoch, ckpt_name))

    return cnn_model  # Return the final model

def test(cnn_model, real_test_loader):
    labels = []
    ids = []
    cnn_model.eval()
    with torch.no_grad():
        for batch_idx, (images, image_name) in enumerate(real_test_loader):
            images = images.to(device)
            outputs = cnn_model(images)
            prob = torch.nn.functional.softmax(outputs, dim=1).data.tolist()
            _, predicted = torch.max(outputs.data, 1)
            print("batch %d/%d" % (batch_idx, len(real_test_loader)))
            for name in image_name:
                ids.append(os.path.basename(name).split('.')[0])
            predicted = predicted.data.tolist()
            for item in predicted:
                labels.append(item)
    submission = pd.DataFrame({'id': ids, 'label': labels})
    output_file_name = "submission.csv"
    submission.to_csv(output_file_name, index=False)
    print("Submission saved at %s" % output_file_name)

def main():
    if len(sys.argv) < 3:
        print("Error: usage: python main.py train/test!")
        exit(0)
    else:
        mode = sys.argv[2]
    print("Mode:", mode)

    # Data augmentation for training as specified:
    # 4-pixel padding on each side, random 32x32 crop, and random horizontal flip.
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])

    # For testing, use the original 32x32 image.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
    ])

    # Create model
    model = ResNet(layer_n).to(device)
    # Use DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("Model created!")

    start_epoch = 0
    lr = 0.1

    # Resume model if checkpoint exists
    if os.path.exists(ckpt_name):
        status_dict = torch.load(ckpt_name)
        model.load_state_dict(status_dict["state"])
        start_epoch = status_dict["epoch"]
        lr = status_dict["lr"]
        print("Resuming model from checkpoint at epoch %d" % start_epoch)

    if mode == 'train':
        # Use torchvision's CIFAR10 dataset
        train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # Train the model until 64k iterations are reached.
        model = train(model, start_epoch, train_loader, test_loader, lr, auto_lr=True)

        # Save the final model to a separate file.
        final_ckpt = ckpt_name.replace('.pth', '_final.pth')
        torch.save(model.state_dict(), final_ckpt)
        print("Final model saved at %s" % final_ckpt)

    elif mode == 'test':
        test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        test(model, test_loader)

if __name__ == "__main__":
    main()
