import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch import utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import copy
import os
import errno
from collections import OrderedDict


class imageclassifier:
    def __init__(self, arch, hidden_units, learning_rate):
        self.architecture = arch
        self.hidden_units = hidden_units
        self.lr = learning_rate

    def build_model(self):
        md = self.architecture
        model = getattr(models, md)(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        if md == 'vgg16' or md == 'vgg19':
            in_features = model.classifier[0].in_features
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(in_features, 500)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.1)),
                                    ('fc2', nn.Linear(500, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=self.lr)
        elif md == 'resnet18' or md == 'resnet34':
            in_features = model.fc.in_features
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(in_features, 500)),
                                    ('relu1', nn.ReLU()),
                                    ('dropout1', nn.Dropout(0.1)),
                                    ('fc2', nn.Linear(500, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))
            model.fc = classifier
            optimizer = optim.Adam(model.fc.parameters(), lr=self.lr)
        else:
            print('Please choose the model from: vgg16, vgg19, resnet18, resneet34')
            
        return model, optimizer, classifier

    
    def train_model(self,model, image_datasets, criterion, optimizer, scheduler, gpu, num_epochs=10):
        epochs = num_epochs
        best_acc = 0
        
        #Using the image datasets and the trainforms, define the dataloaders
        data_loaders = {x:utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
        #Get the size of train, valid, and test sets
        dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        
        # change to cuda
        if gpu:
            model.to('cuda')

        #copy the current best model weights
        best_model_wts = copy.deepcopy(model.state_dict())

        for e in range(epochs):
            print('Epoch: ' + str(e+1) +'/' + str(epochs))
            print('-----------')
            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0
                running_correct = 0

                for inputs, labels in data_loaders[phase]:
                    if gpu:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_correct += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_correct.double() / dataset_sizes[phase]

                print(phase + '   ' + 'Loss:{:.4f} Acc: {:.2%}'.format(epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
           
            print()

        print('Best Validation Acc: {:.2%}'.format(best_acc))
        model.load_state_dict(best_model_wts)
    
        #Load class to indices
        model.class_to_idx = image_datasets['train'].class_to_idx
        
        return model

    
    def save_model(self, model, arch, classifier, optimizer, num_epochs, save_dir):

        checkpoint = {
            'arch': arch,
            'classifier': classifier,
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'num_epoch': num_epochs + 1,
            'class_to_idx': model.class_to_idx
        }
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        
        save_path = os.path.join(save_dir, 'checkpoint.pth')
        torch.save(checkpoint, save_path)
        
       