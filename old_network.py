import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import argparse
from torch.utils.tensorboard import SummaryWriter



#classes = ['Malignant', 'CAF', 'Endothelial', 'B Cell', 'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil', 'NK', 'Plasma',\
#       'T CD4', 'T CD8', 'Unidentifiable', 'B cell exhausted', 'B cell naive', 'B cell non-switched memory', \
#       'B cell switched memory', 'cDC1 CLEC9A', 'cDC2 CD1C', 'cDC3 LAMP3', 'Macrophage M1', 'Macrophage M2', \
#       'Macrophage other', 'T CD4 naive', 'Tfh', 'Th1', 'Th17', 'Th2', 'Treg', 'T CD8 central memory', 'T CD8 effector',\
#       'T CD8 effector memory', 'T CD8 exhausted', 'T CD8 naive']
classes = ['Malignant', 'CAF', 'Endothelial', 'B Cell', 'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil', 'NK', 'Plasma',\
         'T CD4', 'T CD8']


class Image_Dataset(torch.utils.data.Dataset):

        def __init__(self, dataset, labels, transforms):
                self.data = np.transpose(dataset, (0, 3, 1, 2))
                self.labels = labels
                self.transforms = transforms

        def __len__(self):
                return len(self.data)

        def __getitem__(self, idx):
                return self.transforms(self.data[idx].float()), self.labels[idx]


class ST_Classifier(nn.Module):
        def __init__(self, num_classes, fine_tune):
                super(ST_Classifier, self).__init__()
                self.encoder = models.resnet18(pretrained=True)
                self.encoder.fc = nn.Identity()
                if not fine_tune:
                        for param in self.encoder.parameters():
                                param.requires_grad = False
                self.relu = nn.ReLU()
                self.fc1 = nn.Linear(512, 128)
                self.fc2 = nn.Linear(128,64)
                self.out = nn.Linear(64,num_classes)
                self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
                x = self.encoder(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.out(x)
                x = self.softmax(x)
                return x


def get_dataloaders(batch_size, shuffle):
        train_data = torch.load("dataset.pt")
        train_labels = torch.load("labels.pt")
        val_data = torch.load("validation_dataset.pt")
        val_labels = torch.load("validation_labels.pt")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])

        train_dataset = Image_Dataset(train_data, train_labels, normalize)
        val_dataset = Image_Dataset(val_data, val_labels, normalize)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_dataloader, val_dataloader



def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, margin, reduction, writer):

        model.train()

        train_loss_per_class = np.zeros((len(classes), epochs))
        test_loss_per_class = np.zeros((len(classes), epochs))
        train_acc_per_class = np.zeros((len(classes), epochs))
        test_acc_per_class = np.zeros((len(classes), epochs))
        for epoch in range(epochs):

                print("\nEpoch " + str(epoch) + "/" + str(epochs - 1))
                print('----------')

                total = 0.0
                correct_per_class = np.zeros(len(classes))
                running_loss = 0.0
                running_loss_per_class = np.zeros((len(classes)), dtype=float)
                for i, data in enumerate(train_dataloader, 0):
                        inputs, labels = data[0].to(device).float(), data[1].to(device).float()
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        total += len(outputs)
                        for j in range(len(classes)):
                                class_outputs = outputs[:, j]
                                class_labels = labels[:, j]
                                class_loss = criterion(class_outputs, class_labels)
                                running_loss_per_class[j] += class_loss
                                for k in range(len(class_outputs)):
                                        if ((class_outputs[k] - class_labels[k])**2) <= margin: correct_per_class[j] += 1
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                print("[" + str(i) + "] loss: " + str(running_loss / total))
                running_loss = 0.0


                (test_loss, test_acc) = test_model(model, val_dataloader, margin, False, reduction)
                for j in range(len(running_loss_per_class)):
                        #train_loss_per_class[j, epoch] = running_loss_per_class[j]
                        writer.add_scalar("Loss/train/" + classes[j], running_loss_per_class[j], epoch)
                        #test_loss_per_class[j, epoch] = test_loss[j]
                        writer.add_scalar("Loss/test/" + classes[j], test_loss[j], epoch)
                        #test_acc_per_class[j, epoch] = test_acc[j]
                        writer.add_scalar("Accuracy/test/" + classes[j], test_acc[j], epoch)
                        #train_acc_per_class[j, epoch] = correct_per_class[j]/total
                        writer.add_scalar("Accuray/train/" + classes[j], correct_per_class[j]/total, epoch)
        '''
        for j in range(len(classes)):
                plt.figure()
                plt.plot(train_loss_per_class[j], "-b", label='train')
                plt.plot(test_loss_per_class[j], "-r", label='test')
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.legend(loc="upper right")
                plt.title(classes[j] + " Loss")
                plt.savefig("loss_curves/" + classes[j] + "_train_test")
                plt.close()
                plt.figure()
                plt.plot(train_acc_per_class[j], "-b", label='train')
                plt.plot(test_acc_per_class[j], "-r", label='test')
                plt.xlabel("epochs")
                plt.ylabel("accuracy")
                plt.legend(loc="lower right")
                plt.title(classes[j] + " Accuracy")
                plt.savefig("acc_curves/" + classes[j] + "_train_test")
                plt.close()
        '''
        print('finished training')
        return model, writer



def test_model(model, val_dataloader, margin, to_print, reduction):
        model.eval()
        correct = 0
        class_correct = np.zeros(len(classes))
        total = 0
        criterion = nn.MSELoss(reduction=reduction)
        class_loss = np.zeros((len(classes)), dtype=float)

        with torch.no_grad():
                for i, data in enumerate(val_dataloader, 0):
                        inputs, labels = data[0].to(device).float(), data[1].to(device).float()
                        outputs = model(inputs)
                        total += outputs.size(0)
                        for j in range(len(outputs)):
                                if ((outputs[j] - labels[j])**2).sum()/len(classes) <= margin: correct += 1
                                for k in range(len(outputs[j])):
                                        if ((outputs[j, k] - labels[j, k])**2) <= margin: class_correct[k] += 1
                        for j in range(len(classes)):
                                loss = criterion(outputs[:, j], labels[:, j])
                                class_loss[j] += loss
        if to_print:
                print("Accuracy on validation set: " + str(correct/total))
                for i in range(len(classes)):
                        print("Accuracy on " + classes[i] + ": " + str(class_correct[i]/total))

        return class_loss, class_correct/total
      
def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--shuffle', type=bool)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--momentum', type=float)
        parser.add_argument('--margin', type=float)
        parser.add_argument('--fine_tune', type=bool)
        parser.add_argument('--reduction', type=str)

        return parser.parse_args()


if __name__ == '__main__':

        #get input arguments
        args = get_args()
        batch_size = args.batch_size
        shuffle = args.shuffle
        epochs = args.epochs
        lr = args.lr
        momentum = args.momentum
        margin = args.margin
        fine_tune = args.fine_tune
        reduction = args.reduction


        print("\n\n\n")
        print(batch_size)
        print(shuffle)
        print(epochs)
        print(lr)
        print(momentum)
        print(margin)
        print(fine_tune)
        print(reduction)



        #create writer
        writer = SummaryWriter(comment='_' + str(batch_size) + '_' + str(lr) + '_' + str(fine_tune))

        #create model
        model = ST_Classifier(len(classes), fine_tune)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        #create dataloaders
        (train_dataloader, val_dataloader) = get_dataloaders(batch_size, shuffle)

        #set criterion and optimizer
        criterion = nn.MSELoss(reduction=reduction)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        #train model
        (model, writer) = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, margin, reduction, writer)

        writer.flush()
        writer.close()
        #test model
        test_model(model, val_dataloader, margin, True, reduction)
