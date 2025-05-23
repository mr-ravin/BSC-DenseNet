import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import numpy as np
import os
import random
from dataloader import CIFAR100_Dataset
from densenet import get_densenet_models
from utils import plot_graph

# Note: To see better comparison of DenseNet vs BSC-DenseNet, disable the StepLR by setting --use_scheduler to False

parser = argparse.ArgumentParser(description = "Pytorch BSC-DENSENET 121 model for multi-class classification.")
parser.add_argument('-lr_densenet', '--learning_rate_densenet', default = 4e-3) # worked better for Densenet
parser.add_argument('-lr_bsc_densenet', '--learning_rate_bsc_densenet', default = 4e-3) # worked better for BSC Densenet
parser.add_argument('-sch', '--use_scheduler', default="True", help="Use StepLR scheduler during training")
parser.add_argument('-dim','--dim', default=32)
parser.add_argument('-ep', '--epoch', default = 30)
parser.add_argument('-m', '--mode', default="train")
parser.add_argument('-c', '--num_classes', default=100)
parser.add_argument('-d', '--device', default="cuda")
args = parser.parse_args()
LR_DENSENET = args.learning_rate_densenet
LR_BSC_DENSENET = args.learning_rate_bsc_densenet
DIM = args.dim
EPOCH = int(args.epoch)
MODE = args.mode
DEVICE = args.device
NUM_CLASSES = int(args.num_classes)
USE_SCHEDULER = args.use_scheduler.lower()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

if not os.path.isdir("./models"):
    os.mkdir("./models")

if USE_SCHEDULER == "true":
    print("--use_scheduler is enabled.")
else:
    print("--use_scheduler is disabled.")

root_path = "./"
DenseNet, BSC_DenseNet = get_densenet_models(NUM_CLASSES) # returns both Densenet-121 and BSC-Densenet-121 models So that we can compare on CIFAR 100

if DEVICE != "cpu":
    DenseNet.to(DEVICE)
    BSC_DenseNet.to(DEVICE)

def label_wise_accuracy(output, target):
    correct = 0
    incorrect = 0
    for idx in range(output.shape[0]):
        out_class = torch.argmax(output[idx])
        label_class = target[idx]
        if out_class == label_class:
            correct += 1
        else:
            incorrect += 1
    label_accuracy = correct/(correct+incorrect)
    return label_accuracy


def train(total_epoch):
    loss_fn = nn.CrossEntropyLoss()
    densenet_optimizer = optim.Adam(DenseNet.parameters(), lr=LR_DENSENET)
    bsc_densenet_optimizer = optim.Adam(BSC_DenseNet.parameters(), lr=LR_BSC_DENSENET)
    if USE_SCHEDULER != "false":
        densenet_scheduler = StepLR(densenet_optimizer, step_size=3, gamma=0.5)
        bsc_densenet_scheduler = StepLR(bsc_densenet_optimizer, step_size=3, gamma=0.5)
    # Not using data-augmentation
    transform_train = A.Compose([
        A.Resize(height=DIM, width=DIM),
        A.RandomCrop(height=DIM, width=DIM),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),  # Reduced angle to avoid distortion
        A.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
        ToTensorV2(),
        ])
    transform_test = A.Compose([
        A.Resize(height=DIM, width=DIM),
        A.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
        ToTensorV2(),
    ])
    training_data = CIFAR100_Dataset(transform_train, device = DEVICE, mode="train")
    test_data = CIFAR100_Dataset(transform_test, device = DEVICE, mode="test")
    train_dataloader = DataLoader(training_data, batch_size=120, shuffle=True, pin_memory=True if DEVICE == "cpu" else False)
    test_dataloader = DataLoader(test_data, batch_size=120, shuffle=True, pin_memory=True if DEVICE == "cpu" else False)
    densenet_epoch_tr_loss, densenet_epoch_vl_loss = [],[]
    densenet_epoch_tr_acc, densenet_epoch_vl_acc = [], []
    bsc_densenet_epoch_tr_loss, bsc_densenet_epoch_vl_loss = [],[]
    bsc_densenet_epoch_tr_acc, bsc_densenet_epoch_vl_acc = [], []
    densenet_valid_loss_min, bsc_densenet_valid_loss_min = np.inf, np.inf
    densenet_valid_acc_max, bsc_densenet_valid_acc_max = -1, -1
    for ep in range(total_epoch):
        densenet_train_acc, bsc_densenet_train_acc = 0.0, 0.0
        densenet_valid_acc, bsc_densenet_valid_acc = 0.0, 0.0
        train_batch_run, valid_batch_run  = 0, 0
        densenet_train_losses, bsc_densenet_train_losses = [], []
        densenet_valid_losses, bsc_densenet_valid_losses = [], []
        DenseNet.train()
        BSC_DenseNet.train()
        with tqdm(train_dataloader, unit=" Train batch") as tepoch:
            tepoch.set_description(f"Train Epoch {ep+1}")
            for input_images, gt_labels in tepoch:
                train_batch_run += 1
                densenet_optimizer.zero_grad()
                bsc_densenet_optimizer.zero_grad()
                densenet_ouput = DenseNet(input_images)
                bsc_densenet_output = BSC_DenseNet(input_images)
                
                densenet_loss = loss_fn(densenet_ouput, gt_labels)
                bsc_densenet_loss = loss_fn(bsc_densenet_output, gt_labels)
                densenet_train_losses.append(densenet_loss.item())
                bsc_densenet_train_losses.append(bsc_densenet_loss.item())

                densenet_accuracy_value = label_wise_accuracy(densenet_ouput, gt_labels)
                bsc_densenet_accuracy_value = label_wise_accuracy(bsc_densenet_output, gt_labels)
                
                densenet_train_acc += densenet_accuracy_value*100
                bsc_densenet_train_acc += bsc_densenet_accuracy_value*100
                densenet_loss.backward()
                bsc_densenet_loss.backward()
                densenet_optimizer.step()
                bsc_densenet_optimizer.step()
                

        DenseNet.eval()
        BSC_DenseNet.eval()
        with tqdm(test_dataloader, unit=" Valid batch") as vepoch:
            vepoch.set_description(f"Valid Epoch {ep+1}")
            for input_images, gt_labels in vepoch:
                valid_batch_run += 1
                with torch.no_grad():
                    densenet_ouput = DenseNet(input_images)
                    bsc_densenet_output = BSC_DenseNet(input_images)
                    densenet_loss = loss_fn(densenet_ouput, gt_labels)
                    bsc_densenet_loss = loss_fn(bsc_densenet_output, gt_labels)
                densenet_valid_losses.append(densenet_loss.item())
                bsc_densenet_valid_losses.append(bsc_densenet_loss.item())
                densenet_accuracy_value = label_wise_accuracy(densenet_ouput, gt_labels)
                bsc_densenet_accuracy_value = label_wise_accuracy(bsc_densenet_output, gt_labels)
                densenet_valid_acc += densenet_accuracy_value*100
                bsc_densenet_valid_acc += bsc_densenet_accuracy_value*100
                

        # matrices log
        densenet_epoch_train_loss = np.mean(densenet_train_losses)
        bsc_densenet_epoch_train_loss = np.mean(bsc_densenet_train_losses)

        densenet_epoch_val_loss = np.mean(densenet_valid_losses)
        bsc_densenet_epoch_val_loss = np.mean(bsc_densenet_valid_losses)

        densenet_epoch_train_acc = round(densenet_train_acc/train_batch_run,2)
        bsc_densenet_epoch_train_acc = round(bsc_densenet_train_acc/train_batch_run,2)


        densenet_epoch_val_acc = round(densenet_valid_acc/valid_batch_run,2)
        bsc_densenet_epoch_val_acc = round(bsc_densenet_valid_acc/valid_batch_run,2)

        # Logging data 
        densenet_epoch_tr_loss.append(densenet_epoch_train_loss)
        bsc_densenet_epoch_tr_loss.append(bsc_densenet_epoch_train_loss)
        densenet_epoch_vl_loss.append(densenet_epoch_val_loss)
        bsc_densenet_epoch_vl_loss.append(bsc_densenet_epoch_val_loss)

        densenet_epoch_tr_acc.append(densenet_epoch_train_acc)
        bsc_densenet_epoch_tr_acc.append(bsc_densenet_epoch_train_acc)
        densenet_epoch_vl_acc.append(densenet_epoch_val_acc)
        bsc_densenet_epoch_vl_acc.append(bsc_densenet_epoch_val_acc)
        print(f'Epoch {ep+1}')
        print("DENSENET: ")
        print(f'train_loss : {densenet_epoch_train_loss} val_loss : {densenet_epoch_val_loss}')
        print(f'train_accuracy : {densenet_epoch_train_acc} val_accuracy : {densenet_epoch_val_acc}')
        print("BSC-DENSENET: ")
        print(f'train_loss : {bsc_densenet_epoch_train_loss} val_loss : {bsc_densenet_epoch_val_loss}')
        print(f'train_accuracy : {bsc_densenet_epoch_train_acc} val_accuracy : {bsc_densenet_epoch_val_acc}')
        if densenet_epoch_val_loss <= densenet_valid_loss_min or densenet_valid_acc_max <= densenet_epoch_val_acc:
            os.system("rm ./models/densenet_*.pth")
            print("Densenet: removing stored weights of previous epoch")
            torch.save(DenseNet.state_dict(), root_path+"models/densenet_"+str(ep+1)+".pth")
            print('Densenet: Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(densenet_valid_loss_min, densenet_epoch_val_loss))
            if densenet_epoch_val_loss < densenet_valid_loss_min:
                densenet_valid_loss_min = densenet_epoch_val_loss
            if densenet_valid_acc_max < densenet_epoch_val_acc:
                densenet_valid_acc_max = densenet_epoch_val_acc
        
        if bsc_densenet_epoch_val_loss <= bsc_densenet_valid_loss_min or bsc_densenet_valid_acc_max <= bsc_densenet_epoch_val_acc:
            os.system("rm ./models/bsc_densenet_*.pth")
            print("BSC-Densenet: removing stored weights of previous epoch")
            torch.save(BSC_DenseNet.state_dict(), root_path+"models/bsc_densenet_"+str(ep+1)+".pth")
            print('BSC-Densenet: Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(bsc_densenet_valid_loss_min, bsc_densenet_epoch_val_loss))
            if bsc_densenet_epoch_val_loss <= bsc_densenet_valid_loss_min:
                bsc_densenet_valid_loss_min = bsc_densenet_epoch_val_loss
            if bsc_densenet_valid_acc_max <= bsc_densenet_epoch_val_acc:
                bsc_densenet_valid_acc_max = bsc_densenet_epoch_val_acc
        
        if USE_SCHEDULER != "false":
            densenet_scheduler.step()
            bsc_densenet_scheduler.step()

    x_data = [i for i in range(1,EPOCH+1)]
    plot_graph(root_path, x_data, densenet_epoch_tr_loss, bsc_densenet_epoch_tr_loss, densenet_epoch_vl_loss, bsc_densenet_epoch_vl_loss, densenet_epoch_tr_acc, bsc_densenet_epoch_tr_acc, densenet_epoch_vl_acc, bsc_densenet_epoch_vl_acc)

if MODE == "train":
    train(EPOCH)
