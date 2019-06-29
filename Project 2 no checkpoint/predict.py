import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json


def input_args():
    parser = argparse.ArgumentParser(description='Inputs for image classifier')

    parser.add_argument('image_path', type=str, default='flowers/test/1/image_06743.jpg', help='Directory for image')
    parser.add_argument('checkpoint', type=str, default='model.pth', help='checkpoint with trained image classifier.')
    parser.add_argument('--top_k', type=int, default=3, help='Desired number of top predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')

    return parser.parse_args()


def process_image():
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Tensor
    '''
    arg = input_args()
    image = arg.image_path

    im = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    return transform(im)


def imshow(image, ax=None, title=None,):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict():
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    arg = input_args()
    image_path = arg.image_path
    model = arg.checkpoint
    topk = arg.top_k
    device = arg.device

    image = process_image()
    image = image.unsqueeze(0)

    checkpoint = torch.load(model)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model.state_dict'])
    model.class_to_idx = checkpoint['model.class_to_idx']
    class_to_idx = {ii: i for i, ii in model.class_to_idx.items()}

    #device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    image, model = image.to(device), model.to(device)

    with torch.no_grad():
        model.eval()
        output = model.forward(image)

        out_prob = torch.exp(output)
        probs, classes = out_prob.topk(topk, dim=1)
        probs, classes = probs.to('cpu'), classes.to('cpu')

    probs1, label = probs.numpy(), classes.numpy()

    classes = []
    probs = []

    for i, ii in np.ndenumerate(label):
        classes.append(class_to_idx[int(ii)])
    classes = np.array(classes)
    for i, ii in np.ndenumerate(probs1):
        probs.append(ii)
    probs = np.array(probs)

    return probs, classes


def show_predict():

    arg = input_args()
    category_names = arg.category_names

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict()
    image = process_image()
    names = []
    for i in classes:
        names.append(cat_to_name[i])
    tuple(names)

    print("Top Prediction \nName: {} .. \nProbability: {}".format(names, probs))

    #ax1 = imshow(image)
    #ax1.set_title(cat_to_name[classes[0]])

    #fig, ax2 = plt.subplots()
    #y_pos = np.arange(len(names))
    #ax2.barh(y_pos, probs)
    #ax2.set_yticks(y_pos)
    #ax2.set_yticklabels(names)
    #ax2.invert_yaxis()

    return


if __name__ == '__main__':

    show_predict()