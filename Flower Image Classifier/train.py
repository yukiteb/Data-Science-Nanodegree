import argparse
import c_imageclassifier
import flower_utils
from torchvision import datasets
import os
from torch import utils
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir",help="data directory")
    parser.add_argument("-s", "--save_dir", dest="save_dir",default="./",help="Set directory to save checkpoints")
    parser.add_argument("-a", "--arch", dest="arch", default ="vgg16", help="Set the architecture: vgg16, vgg19, resnet18, resnet34")
    parser.add_argument("-l", "--learning_rate", dest="learning_rate", type=float, default =0.001, help="Set the learning rate")
    parser.add_argument("-u", "--hidden_units", dest="hidden_units", type=int, default =512, help="Set the number of hidden units")
    parser.add_argument("-e", "--epochs", dest="epochs",type=int, default =5, help="Set the number of epochs")
    parser.add_argument("-g", "--gpu", dest="gpu",action="store_true", help="Enable GPU mode")
    
    #Get arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    num_epochs = args.epochs
    gpu = args.gpu
    
    #Load the datasets with ImageFolder
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),flower_utils.data_transforms[x]) for x in ['train', 'valid', 'test']}
    
    #Build the model
    img_classifier = c_imageclassifier.imageclassifier(arch, hidden_units,lr)
    model, optimizer, classifier = img_classifier.build_model()
    
    #Set criterion, optimizer
    criterion = nn.NLLLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    #Train the model
    if gpu:
        print('Starting training on GPU mode!')
    trained_model = img_classifier.train_model(model, image_datasets, criterion, optimizer, scheduler, gpu,num_epochs)
    
    #Save the model
    img_classifier.save_model(trained_model, arch, classifier, optimizer, num_epochs, save_dir)
    

if __name__ == "__main__":
    main()

