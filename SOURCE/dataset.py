from PIL import Image
import PIL
import os
import torchvision.utils
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import random

def read_file(path):
    
    """Read files that have the split information"""
    
    with open(path,'r') as split_info:
        images_ = split_info.readlines()
    return images_


def set_label(samples, label=None):
    
    """
    Assign labels to the cases, COVID label is 1, nonCOVID is 0
    """
    
    if label == 'CT_COVID':
        label = 1
    else:
        label = 0
        
    labels = []
    for i in range(len(samples)):
        labels.append(label)
        
    return labels


class CT_Triplet(Dataset):
    
    """
    Args: root_dir -> The root directory where the Dataset is found
          split_covid -> Name of the file with covid images split instruction
          split_nonCovid -> Name of the file with non covid images split instruction
          trnsform -> Default None but can take as imput the transformer defined earlier

    @getPositive: Returns a positive sample for the anchor selected randomly from the dataset
    @getNegative: Returns a negative sample for the anchor selected randomly from the dataset
    
    Outpu: __len__ -> The length of the set of data
           __getitem__ -> returns a triplete (preprocessed if transformer is not NONE) 
                          composed of anchor, positive and negative sample together with their corresponding labels
                        
    """
    
    def __init__(self, root_dir, split_covid, split_nonCovid, transform=None):
        
        self.root_dir = root_dir
        self.split_covid = split_covid
        self.split_nonCovid = split_nonCovid
        self.transform = transform
        self.labels_ = ['CT_NonCOVID','CT_COVID']
        
        covid_list = read_file(os.path.join(self.root_dir, 'CT_COVID',self.split_covid))
        nonCovid_list  = read_file(os.path.join(self.root_dir, 'CT_NonCOVID',self.split_nonCovid))
        self.img_list = covid_list + nonCovid_list
        
        CovidLabels = set_label(covid_list, 'CT_COVID')
        NonCovidLabels = set_label(nonCovid_list)
        self.labels = CovidLabels + NonCovidLabels
        
    def __len__(self):
        return len(self.img_list)
    
    
    def getPositive(self, idx, curr_label):
        
        if curr_label == 0:
            label = 1
        else:
            label = 0        
        
        while(label != curr_label):
            label_rand = random.choice(self.img_list)
            index = self.img_list.index(label_rand)
            if(index != idx):
                label = self.labels[index]
                
        return index
        
        
    def getNegative(self, idx, curr_label):
        
        label = curr_label        
        
        while(label == curr_label):
            image = random.choice(self.img_list)
            index = self.img_list.index(image)
            label = self.labels[index]
                
        return index
    
    def __getitem__(self, idx):
            
        image_path = os.path.join(self.root_dir,self.labels_[int(self.labels[idx])],self.img_list[idx].rstrip())
        image_a = Image.open(image_path).convert('RGB')            
        label = self.labels[idx]
        
        
        idx_p = self.getPositive(idx, label)
        image_path_p = os.path.join(self.root_dir,self.labels_[int(self.labels[idx_p])],self.img_list[idx_p].rstrip())
        image_p = Image.open(image_path_p).convert('RGB')
        label_p = self.labels[idx_p]
        
        idx_n = self.getNegative(idx, label)
        image_path_n = os.path.join(self.root_dir,self.labels_[int(self.labels[idx_n])],self.img_list[idx_n].rstrip())
        image_n = Image.open(image_path_n).convert('RGB')
        label_n = self.labels[idx_n]
        
        # print("anchor :", label)
        # print("positive",label_p)
        # print("negative", label_n)
        
                
        if self.transform != None:
            # image_a = self.anchor_transform(image_a)
            image_a = self.transform(image_a)
            image_p = self.transform(image_p)
            image_n = self.transform(image_n)
            
        
        return image_a, image_p, image_n , label, label_p, label_n



class CT_Images(Dataset): #single image batches
    
    """
    Args: root_dir -> The root directory where the Dataset is found
          split_covid -> Name of the file with covid images split instruction
          split_nonCovid -> Name of the file with non covid images split instruction
          trnsform -> Default None but can take as imput the transformer defined earlier
    
    Outpu: __len__ -> The length of the set of data
           __getitem__ -> returns an image (preprocessed if transformer is not NONE) and the corresponding label
    """
    
    def __init__(self, root_dir, split_covid, split_nonCovid, transform=None):
        
        self.root_dir = root_dir
        self.split_covid = split_covid
        self.split_nonCovid = split_nonCovid
        self.transform = transform
        self.labels_ = ['CT_NonCOVID','CT_COVID']
        
        covid_list = read_file(os.path.join(self.root_dir, 'CT_COVID',self.split_covid))
        nonCovid_list  = read_file(os.path.join(self.root_dir, 'CT_NonCOVID',self.split_nonCovid))
        self.img_list = covid_list + nonCovid_list
        
        CovidLabels = set_label(covid_list, 'CT_COVID')
        NonCovidLabels = set_label(nonCovid_list)
        self.labels = CovidLabels + NonCovidLabels
        
        
    def __len__(self):
        return len(self.img_list)
    
        
    def __getitem__(self, idx):
            
        image_path = os.path.join(self.root_dir,self.labels_[int(self.labels[idx])],self.img_list[idx].rstrip())
        image = Image.open(image_path).convert('RGB')
            
        label = self.labels[idx]
        
        if self.transform != None:
            image = self.transform(image)
        
        return image, label