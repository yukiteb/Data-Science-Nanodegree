import argparse
import c_imageclassifier
import flower_utils

import numpy as np
import pandas as pd
import torch
#from torch import nn
#from torch import optim
#from torch.optim import lr_scheduler
from torch import utils
#import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import sys

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
    except:
        sys.exit('Could not load the checkpoint')
    arch = checkpoint['arch']
    model = getattr(models,arch)(pretrained=True)
    if arch == 'vgg16' or arch == 'vgg19':
        model.classifier = checkpoint['classifier']
    elif arch == 'resnet18' or arch == 'resnet34':
        model.fc = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, gpu, topk=5):
    
    #Open image
    try:
        img = Image.open(image_path)
    except:
        sys.exit('The file ' + image_path + ' does not exist!')
    
    img = flower_utils.process_image(img)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img.unsqueeze_(0)
    
    if gpu:
        img = img.to('cuda')
        model.to('cuda')
    else:
        model.to('cpu')
    
    #Set the model to evaluation mode
    model.eval()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    #Get topk probs and indices
    probs, indices = output.topk(topk)
    probs = torch.exp(probs)
    probs = probs.to('cpu').numpy()
    
    #Get the list of indices
    cat_list_index = np.array(indices[0])
    #Invert from class->category_indices to category_indices->class
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    #Get the topk category indices
    topk_labels = [idx_to_class[x] for x in cat_list_index]
    
    return probs[0], topk_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path",help="image path")
    parser.add_argument("checkpoint", help="check point")
    parser.add_argument("-t","--top_k",dest="top_k", type=int, default = 5, help="Number of top k probabilities")
    parser.add_argument("-c","--category_names",dest="category_names", help="Use a mapping of categories to real names")
    parser.add_argument("-g","--gpu", dest="gpu",action="store_true",help="Enable GPU for inference")
    
    #Get arguments
    args = parser.parse_args()
    img_path = args.img_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name = args.category_names
    gpu = args.gpu
                        
    model = load_checkpoint(checkpoint)
    probs, topk_labels = predict(img_path, model, gpu, top_k)
    
    y_pos = np.arange(top_k)
    
    
    if cat_to_name:
        #Get the names of topk flowers
        cat_to_name_dict = flower_utils.get_cat_to_name(cat_to_name)
        cat_list_name = [cat_to_name_dict[x] for x in topk_labels]
        #plt.yticks(y_pos, cat_list_name)
    else:
        cat_list_name = topk_labels
        #plt.yticks(y_pos,topk_labels)

    #plt.xlabel('Probabilities')

    #plt.barh(y_pos, np.array(probs[0]))
    #plt.show()
    print('Category : Probability')
    for cat, prob in zip(cat_list_name, probs):
        print (cat, '{:.2%}'.format(prob))
    #print('Top ' + str(top_k) + ' probabilities:' + probs)
    #print('Top ' + str(top_k) + ' categories:' + cat_list_name)
    

if __name__ == "__main__":
    main()
