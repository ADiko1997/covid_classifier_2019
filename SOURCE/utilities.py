from PIL import Image
import PIL
import os
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc
from torch.utils.tensorboard import SummaryWriter
import math
import cv2
torch.cuda.empty_cache()

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),0)
    new_image = image_blur
    return new_image

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels,bs, classes):

    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''

    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(50, 50))
    for idx in np.arange(bs):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def evaluate(labels, pred, epoch):
    
    """
    Functionality: Calculate the metrics necesary to evaluate the model (specified from the challange rules)
    
    :labels: Ground truth labels 
    :pred:   Prediction resuls
    :epoch:  Current epoch
    
    Output: accuracy calculated from accuracy score part of sklearn metrics
            F1 calculated again from sklearn metrics
            auc_score again from sklearn metrics
    """
    
    accuracy = accuracy_score(labels.numpy(), pred[0][0])
    F1 = f1_score(labels.numpy(), pred[0][0])
    fpr, tpr, thresholds = roc_curve(labels.numpy(), pred[0][0])
    auc_score = auc(fpr, tpr)
    
    return accuracy, F1, auc_score


def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize = (20, 20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def vizBatch(data_loader, labels, num): 

    "Vizualizes num images from single image dataloader"

    dataiter = iter(data_loader)
    images, labels_ = dataiter.next()
    images_p = images[:num]
    # show images
    imshow(torchvision.utils.make_grid(images_p))
    # print labels
    print(' '.join('%5s' % labels[labels_[j]] for j in range(num)))


def tripViz(data_loader, labels, num):  

    "Visualizes batches of triplets"

    dataiter = iter(data_loader)
    images_a, images_p, images_n, labels_a, labels_p, labels_n = dataiter.next()
    images_a = images_a[:num]
    images_p = images_p[:num]
    images_n = images_n[:num]
    # show images
    imshow(torchvision.utils.make_grid(images_a))
    print(' '.join('%5s' % labels[labels_a[j]] for j in range(num)))
    imshow(torchvision.utils.make_grid(images_p))
    print(' '.join('%5s' % labels[labels_p[j]] for j in range(num)))
    imshow(torchvision.utils.make_grid(images_n))
    print(' '.join('%5s' % labels[labels_n[j]] for j in range(num)))

def set_training_parameters(model, feature_extracting):
    
    """
    Decide which parametters to learn
    .requires_grad is by default true and is useful for finetuning and training from scratch
    .requires_grad is set to false if we want to use the model as feature extractor
    """
    
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, feature_extract, num_classes, pretrained=True):
    
    """
    Initialize the pretrained models, resnet50 and densenet169 were selected as the architectures to try
    Credits to: Pytorch.org documentation
    
    @params:
            model_name: name of the selected model
            feature_extract: a bool that specifies if we want to do fine tuning or feature extract transfer learning
            num_classes: number of classes that our dataset has
            pretrained: by default equal to tru if we want to use the ImageNet weights or only the architecture
            
    """
    
    model_pt = None
        
    if model_name == 'resnet50':
        
        model_pt = torchvision.models.resnet50(pretrained=pretrained).cuda()
        set_training_parameters(model_pt, feature_extract)
        num_features = model_pt.fc.in_features
        model_pt.fc = torch.nn.Linear(num_features, num_classes)
    
    elif model_name == 'resnet18':
        
        model_pt = torchvision.models.resnet50(pretrained=pretrained).cuda()
        set_training_parameters(model_pt, feature_extract)
        num_features = model_pt.fc.in_features
        model_pt.fc = torch.nn.Linear(num_features, num_classes)
        
    elif model_name == 'densenet169':
        
        # model_pt = torchvision.models.densenet169(pretrained=pretrained).cuda()
        model_pt = torchvision.models.densenet169(pretrained=pretrained)
        set_training_parameters(model_pt, feature_extract)
        num_ftrs = model_pt.classifier.in_features
        model_pt.classifier = torch.nn.Linear(num_ftrs, num_classes)
    
    elif model_name == 'densenet161':
        
        model_pt = torchvision.models.densenet161(pretrained=pretrained).cuda()
        set_training_parameters(model_pt, feature_extract)
        num_ftrs = model_pt.classifier.in_features
        model_pt.classifier = torch.nn.Linear(num_ftrs, num_classes)

        
    else:
        print("Invalid name, specify the model correctly")
        
    return model_pt


def transformers(normalize):

    train_transformer = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop((224,224)), #224 is the size of the imagenet images so this will allow to use transfer learning
    transforms.RandomHorizontalFlip(p=0.7), #Randomly horziontal flip image with probability p, default p = 0.5
    transforms.RandomVerticalFlip(p=0.7), #Randomly vertical flip image with probability p, default p = 0.5
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #randomly changes the brightness and contrast based on a given formula
    transforms.ToTensor(),#Transforms the image to pytorch tensor (by default the values are in converted in range [0,1])
    normalize
    ])

    val_transformer = transforms.Compose([ #This transformer will be used even for the test set
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])

    return train_transformer, val_transformer




