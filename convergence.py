# #coding:utf-8

# import os
# import sys
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from data import *   # Keep any other data imports if needed
# from models import *

# import torchvision
# from torchvision import transforms, utils
# from torchvision.datasets import CIFAR10
# from tensorboardX import SummaryWriter

# import pandas as pd

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# layer_n = int(sys.argv[1])

# # Define checkpoint and log directory names based on the network depth.
# ckpt_name = "checkpoints/ResNet-%d_cifar10.pth" % (layer_n*6+2)
# log_name = "./logs/ResNet-%d_cifar10_log/" % (layer_n*6+2)

# # Ensure the directories exist
# os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
# os.makedirs(log_name, exist_ok=True)

# # Data path for CIFAR10 (will be used for download if not available)
# data_path = '/data4/home/vedantb/datasets/cifar10'
# os.makedirs(data_path, exist_ok=True)

# batch_size = 100

# def evaluate_val(model, val_loader):
#     """Evaluate the model on the validation set and return the accuracy."""
#     model.eval()
#     correct = 0
#     total = 0
#     criterion = nn.CrossEntropyLoss()
#     loss_sum = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             loss_sum += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     val_acc = correct / total
#     avg_loss = loss_sum / len(val_loader)
#     model.train()
#     return val_acc, avg_loss

# def train(cnn_model, start_epoch, train_loader, test_loader, val_loader, lr, auto_lr=True):
#     # Train model from scratch
#     num_epochs = 500

#     learning_rate = lr
#     print("lr: %f" % learning_rate)
#     optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
#     criterion = torch.nn.CrossEntropyLoss()

#     train_writer = SummaryWriter(log_dir=log_name+'train')
#     test_writer = SummaryWriter(log_dir=log_name+'test')

#     # Early stopping parameters
#     best_val_acc = 0.0
#     patience = 20  # number of epochs with no improvement after which training stops
#     patience_counter = 0

#     train_offset = 0
#     train_iter = 0

#     for epc in range(num_epochs):
#         epoch = epc + start_epoch

#         train_total = 0
#         train_correct = 0

#         for batch_idx, (data_x, data_y) in enumerate(train_loader):
#             train_iter = train_offset + epoch * len(train_loader) + batch_idx

#             if auto_lr:
#                 if train_iter == 32000:
#                     learning_rate = learning_rate / 10.
#                     optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate,
#                                                 momentum=0.9, weight_decay=0.0001)
#                 if train_iter == 48000:
#                     learning_rate = learning_rate / 10.
#                     optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate,
#                                                 momentum=0.9, weight_decay=0.0001)
#                 # No break condition now

#             data_x = data_x.to(device)
#             data_y = data_y.to(device)

#             optimizer.zero_grad()
#             output = cnn_model(data_x)
#             loss = criterion(output, data_y)
                        
#             _, predicted = torch.max(output.data, 1)
#             train_total += data_y.size(0)
#             train_correct += (predicted == data_y).sum().item()

#             loss.backward()
#             optimizer.step()

#             if train_iter % 10 == 0:
#                 print("Epoch %d/%d, Step %d/%d, iter %d Loss: %f, lr: %f" \
#                      % (epoch, start_epoch+num_epochs, batch_idx, len(train_loader), train_iter, loss.item(), learning_rate))
#                 train_writer.add_scalar('data/loss', loss, train_iter)

#             if train_iter % 100 == 0:
#                 train_acc = float(train_correct) / train_total
#                 print("iter %d, Train Accuracy: %f" % (train_iter, train_acc))
#                 print("iter %d, Train correct/count: %d/%d" % (train_iter, train_correct, train_total))
#                 train_writer.add_scalar('data/accuracy', train_acc, train_iter)
#                 train_writer.add_scalar('data/error', 1.0 - train_acc, train_iter)
#                 train_total = 0
#                 train_correct = 0

#             if train_iter % 100 == 0:
#                 with torch.no_grad():
#                     correct = 0
#                     total = 0
#                     loss_sum = 0.0
#                     for test_batch_idx, (images, labels) in enumerate(test_loader):
#                         images = images.to(device)
#                         labels = labels.to(device)
#                         outputs = cnn_model(images)
#                         loss_sum += criterion(outputs, labels).item()
#                         _, predicted = torch.max(outputs.data, 1)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()
                    
#                     avg_loss = float(loss_sum) / len(test_loader)
#                     test_writer.add_scalar('data/loss', avg_loss, train_iter)
#                     test_acc = float(correct) / total

#                     print("iter %d, Test Accuracy: %f" % (train_iter, test_acc))
#                     print("iter %d, Test avg Loss: %f" % (train_iter, avg_loss))
#                     test_writer.add_scalar('data/accuracy', test_acc, train_iter)
#                     test_writer.add_scalar('data/error', 1.0 - test_acc, train_iter)

#         # End of epoch: evaluate on validation set
#         val_acc, val_loss = evaluate_val(cnn_model, val_loader)
#         print("Epoch %d: Validation Accuracy: %f, Validation Loss: %f" % (epoch, val_acc, val_loss))
        
#         # Check early stopping condition
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             patience_counter = 0
#             # Save the best model so far
#             best_ckpt = ckpt_name.replace('.pth', '_best.pth')
#             torch.save(cnn_model.state_dict(), best_ckpt)
#             print("New best validation accuracy! Model saved at %s" % best_ckpt)
#         else:
#             patience_counter += 1
#             print("No improvement for %d epoch(s)." % patience_counter)
        
#         if patience_counter >= patience:
#             print("Validation accuracy did not improve for %d epochs. Early stopping." % patience)
#             break

#         # Save a checkpoint at the end of each epoch
#         state_dict = {"state": cnn_model.state_dict(), "epoch": epoch, "acc": test_acc, "lr": learning_rate}
#         torch.save(state_dict, ckpt_name)
#         print("Checkpoint saved! %s" % ckpt_name)
    
#     return cnn_model  # Return the final model

# def test(cnn_model, real_test_loader):
#     labels = []
#     ids = []
#     for batch_idx, (images, image_name) in enumerate(real_test_loader):
#         images = images.to(device)
#         outputs = cnn_model(images)
#         prob = torch.nn.functional.softmax(outputs.data, dim=1).data.tolist()
#         _, predicted = torch.max(outputs.data, 1)
#         print("batch %d/%d" % (batch_idx, len(real_test_loader)))
#         for name in image_name:
#             ids.append(os.path.basename(name).split('.')[0])
#         predicted = predicted.data.tolist()
#         for item in predicted:
#             labels.append(item)
#     submission = pd.DataFrame({'id': ids, 'label': labels})
#     output_file_name = "submission.csv"
#     submission.to_csv(output_file_name, index=False)
#     print("# %s generated!" % output_file_name)

# def weight_init(cnn_model):
#     if isinstance(cnn_model, nn.Linear):
#         nn.init.xavier_normal_(cnn_model.weight)
#         nn.init.constant_(cnn_model.bias, 0)
#     elif isinstance(cnn_model, nn.Conv2d):
#         nn.init.kaiming_normal_(cnn_model.weight, mode='fan_out', nonlinearity='relu')
#     elif isinstance(cnn_model, nn.BatchNorm2d):
#         nn.init.constant_(cnn_model.weight, 1)
#         nn.init.constant_(cnn_model.bias, 0)

# def main():
#     if len(sys.argv) < 3:
#         print("Error: usage: python main.py train/test!")
#         exit(0)
#     else:
#         mode = sys.argv[2]
#     print(mode)

#     # Data augmentation transform for training
#     transform_enhanc_func = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomCrop(32, padding=4, padding_mode='edge'),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.mul(255)),
#         transforms.Normalize([125., 123., 114.], [1., 1., 1.])
#     ])

#     # Transform for validation/testing
#     transform_func = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.mul(255)),
#         transforms.Normalize([125., 123., 114.], [1., 1., 1.])
#     ])

#     # Create model
#     model = ResNet(layer_n).to(device)
#     print("Model created!")

#     start_epoch = 0
#     lr = 0.1

#     # Resume model if checkpoint exists
#     if os.path.exists(ckpt_name):
#         status_dict = torch.load(ckpt_name)
#         model_state = status_dict["state"]
#         start_epoch = status_dict["epoch"] + 1
#         acc = status_dict["acc"]
#         lr = status_dict["lr"]
#         model.load_state_dict(model_state)
#         print("Model loaded from checkpoint!")

#     if mode == 'train':
#         # Use torchvision's CIFAR10 dataset
#         test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform_func)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
#         train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_enhanc_func)
#         # For validation, we use the training split without augmentation.
#         val_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform_func)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
#         val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#         # Train and get the final model with early stopping
#         model = train(model, start_epoch, train_dataloader, test_dataloader, val_dataloader, lr, True)
        
#         # Save the final model to a separate file.
#         final_ckpt = ckpt_name.replace('.pth', '_final.pth')
#         torch.save(model.state_dict(), final_ckpt)
#         print("Final model saved at %s" % final_ckpt)

# if __name__ == "__main__":
#     main()


