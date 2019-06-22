from torchvision import datasets, transforms, models
import numpy as np
import json

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)),
    ]),
    'valid':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)),
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)),
    ])
}


def get_cat_to_name(mapping):
    with open(mapping, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Get the size of the image and compute aspect ratio
    width, height = image.size
    a_ratio = width/height
    
    #Resize the image
    thumb_size = 256, 256
    if width <= height: 
        size = 256, int(256*a_ratio)
    else:
        size = int(256*a_ratio), 256

    image = image.resize(size)
    
    
    width, height = image.size
    crop_width, crop_height = (224,224)
    
    #Crop center 224x224
    left = (width - crop_width)/2
    top = (height - crop_height)/2
    right = (width + crop_width)/2
    bottom = (height + crop_height)/2
    image = image.crop((left, top, right, bottom))
    
    
    #Conver to numpy array
    np_img = np.array(image)
    np_img = np_img/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean.transpose())/std.transpose()
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img