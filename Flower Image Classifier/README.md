# DSND-Flower-Image-Classifier
This project builds a flower image classifier using PyTorch. A few pre-trained network structures are built with pretrained network, and classification layers are trained to classify 102 species of flowers. The model is flexible in that it allows the following options:

- You can build the network with VGG (VGG16/19) or ResNet (Resnet18/34)
- You can change Learning Rate, Number of hidden units, Number of epochs
- You have an option to train the model with GPU

## Installation
Run the following command to clone to local machine

```
git clone https://github.com/yukiteb/DSND-Flower-Image-Classifier.git
```

## File structures

The files are structured as follows:
```
- c_imageclassifier.py # Class containing the image classifier models
- cat_to_name.json # Mapping of category to flower names
- flower_utils.py # Script containing utility functions
- predict.py # Script to make predictions
- train.py # Scrip to train the model
- README.md
```

## How to run
### Training
Basic usage
```
python train.py data_directory
```
Options:
Set directory to save checkpoints:
```
python train.py data_dir --save_dir save_directory
```
Choose architecture:
```
python train.py data_dir --arch "vgg13"
```
Set hyperparameters:
```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```
Use GPU for training:
```
python train.py data_dir --gpu
```

### Prediction

Basic usage:
```
python predict.py /path/to/image checkpoint
```
Options:
Return top K most likely classes:
```
python predict.py input checkpoint --top_k 3
```
Use a mapping of categories to real names:
```
python predict.py input checkpoint --category_names cat_to_name.json
```
Use GPU for inference:
```
python predict.py input checkpoint --gpu
```

