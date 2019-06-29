import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
from collections import OrderedDict
import argparse


def input_args():
    parser = argparse.ArgumentParser(description='Inputs for image classifier')

    parser.add_argument('data_dir', type=str, default='flowers', help='Directory for image folders. Subdirectories must include ./train, ./valid')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture for feature section of image classifier.')
    parser.add_argument('--epochs', type=int, default=20, help='Desired number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Desired learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Total number of hidden layers. Number will be devided to fit network')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--save_dir', type=str, default='model.pth', help='Desired directory for classifier checkpoint')

    return parser.parse_args()

def image_pro():
    '''expects: '/train', '/valid' in data_dir.
    returns: train_loader, valid_loader, test_loader'''

    arg = input_args()
    data_dir = arg.data_dir

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=False)


    return train_loader, train_datasets, valid_loader

def main(train_loader, train_datasets, valid_loader):
    '''args: model_arch, train_loader, train_datasets, valid_loader, epochs, learning_rate, hidden_units, device, save_dir

    saves checkpoint under save_dir

    returns: checkpoint with {'epoch_tot',
                              'model',
                              'criterion',
                              'optimizer',
                              'optimizer.state_dict',
                              'model.state_dict',
                              'model.class_to_idx'}'''

    arg = input_args()
    model_arch = arg.arch
    hidden_units = arg.hidden_units
    learning_rate = arg.learning_rate
    device = arg.device
    epochs = arg.epochs
    save_dir = arg.save_dir

    if model_arch == 'alexnet':
         gs_vgg = models.alexnet(pretrained=True)
    elif model_arch == 'vgg11':
         gs_vgg = models.vgg11(pretrained=True)
    elif model_arch == 'vgg11_bn':
         gs_vgg = models.vgg11_bn(pretrained=True)
    elif model_arch == 'vgg13':
         gs_vgg = models.vgg13(pretrained=True)
    elif model_arch == 'vgg13_bn':
         gs_vgg = models.vgg13_bn(pretrained=True)
    elif model_arch == 'vgg16':
         gs_vgg = models.vgg16(pretrained=True)
    elif model_arch == 'vgg16_bn':
         gs_vgg = models.vgg16_bn(pretrained=True)
    elif model_arch == 'vgg19':
         gs_vgg = models.vgg19(pretrained=True)
    elif model_arch == 'vgg19_bn':
         gs_vgg = models.vgg19_bn(pretrained=True)
    elif model_arch == 'resnet18':
         gs_vgg = models.resnet18(pretrained=True)
    elif model_arch == 'resnet34':
         gs_vgg = models.resnet34(pretrained=True)
    elif model_arch == 'resnet50':
         gs_vgg = models.resnet50(pretrained=True)
    elif model_arch == 'resnet101':
         gs_vgg = models.resnet101(pretrained=True)
    elif model_arch == 'resnet152':
         gs_vgg = models.resnet152(pretrained=True)
    elif model_arch == 'squeezenet1_0':
         gs_vgg = models.squeezenet1_0(pretrained=True)
    elif model_arch == 'squeezenet1_1':
         gs_vgg = models.squeezenet1_1(pretrained=True)
    elif model_arch == 'densenet121':
         gs_vgg = models.densenet121(pretrained=True)
    elif model_arch == 'densenet169':
         gs_vgg = models.densenet169(pretrained=True)
    elif model_arch == 'densenet161':
         gs_vgg = models.densenet161(pretrained=True)
    elif model_arch == 'densenet201':
         gs_vgg = models.densenet201(pretrained=True)
    elif model_arch == 'inception_v3':
         gs_vgg = models.inception_v3(pretrained=True)
    elif model_arch == 'googlenet':
         gs_vgg = models.googlenet(pretrained=True)
    elif model_arch == 'shufflenet_v2_x0_5':
         gs_vgg = models.shufflenet_v2_x0_5(pretrained=True)
    elif model_arch == 'shufflenet_v2_x1_0':
         gs_vgg = models.shufflenet_v2_x1_0(pretrained=True)
    elif model_arch == 'shufflenet_v2_x1_5':
         gs_vgg = models.shufflenet_v2_x1_5(pretrained=True)
    elif model_arch == 'shufflenet_v2_x2_0':
         gs_vgg = models.shufflenet_v2_x2_0(pretrained=True)
    elif model_arch == 'mobilenet_v2':
         gs_vgg = models.mobilenet_v2(pretrained=True)
    elif model_arch == 'resnext50_32x4d':
         gs_vgg = models.resnext50_32x4d(pretrained=True)
    elif model_arch == 'resnext101_32x8d':
         gs_vgg = models.resnext101_32x8d(pretrained=True)
    elif model_arch == 'wide_resnet50_2':
         gs_vgg = models.wide_resnet50_2(pretrained=True)
    elif model_arch == 'wide_resnet101_2':
         gs_vgg = models.wide_resnet101_2(pretrained=True)
    elif model_arch == 'mnasnet0_5':
         gs_vgg = models.mnasnet0_5(pretrained=True)
    elif model_arch == 'mnasnet0_75':
         gs_vgg = models.mnasnet0_75(pretrained=True)
    elif model_arch == 'mnasnet1_0':
         gs_vgg = models.mnasnet1_0(pretrained=True)
    elif model_arch == 'mnasnet1_3':
         gs_vgg = models.mnasnet1_3(pretrained=True)

    epoch_tot = 0


    for parameters in gs_vgg.parameters():
        parameters.requires_grad = False

    try:
        input_layer = gs_vgg.classifier[0].in_features
    except:
        input_layer = gs_vgg.classifier.in_features
    hidden_layers = [(int(hidden_units * 0.68)), (int(hidden_units * 0.32))]
    output_layer = len(train_loader)


    gs_vgg.classifier = nn.Sequential(nn.Linear(input_layer, hidden_layers[0]),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.3),
                                      nn.Linear(hidden_layers[0], hidden_layers[1]),
                                      nn.ReLU(),
                                      #nn.Linear(hidden_layers[1], output_layer),
                                      nn.Linear(hidden_layers[1], 102),
                                      nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(gs_vgg.classifier.parameters(), lr=learning_rate)

    gs_vgg.to(device)
    step_num = 0
    epoch = epochs
    running_loss = 0
    print_every = 10
    for e in range(epoch):
        epoch_tot += 1
        for images, labels in train_loader:

            gs_vgg.train()

            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()

            output = gs_vgg.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            step_num += 1

            if step_num % print_every == 0:
                test_loss = 0
                accuracy = 0
                gs_vgg.eval()
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    output = gs_vgg.forward(images)
                    loss = criterion(output, labels)

                    test_loss += loss.item()

                    prob = torch.exp(output)
                    top_p, top_class = prob.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Total Epochs: {epoch_tot}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {(accuracy/len(valid_loader))*100:.1f}%")
                running_loss = 0
                gs_vgg.train()
    gs_vgg.class_to_idx = train_datasets.class_to_idx
    gs_checkpoint = {'epoch_tot': epoch_tot,
                     'model': gs_vgg,
                     'criterion': criterion,
                     'optimizer': optimizer,
                     'optimizer.state_dict': optimizer.state_dict(),
                     'model.state_dict': gs_vgg.state_dict(),
                     'model.class_to_idx': gs_vgg.class_to_idx}
    torch.save(gs_checkpoint, save_dir)
    return gs_checkpoint

if __name__ == '__main__':
    input_args()
    train_loader, train_datasets, valid_loader = image_pro()
    checkpoint = main(train_loader, train_datasets, valid_loader)
