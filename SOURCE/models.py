import os
import torchvision.utils
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import ImageFile
from PIL import Image
import random
import pickle
#First simple architecture to setup the pipline 
class SimpleCNN(torch.nn.Module):
    
    def __init__(self, input_ch, output):

        super(SimpleCNN, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=input_ch, out_channels=16, kernel_size=[3,3],stride=1, padding=1)
        self.cnn2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3,3],stride=1, padding=1)
        self.cnn3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3,3], padding=1)
        self.cnn4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3,3], padding=1)
        self.cnn5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], padding=1)
        self.cnn6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=[3,3])
        self.Bnorm = torch.nn.BatchNorm2d(num_features=256)
        self.drop = torch.nn.Dropout2d(p=0.3)
        self.lin1 = torch.nn.Linear(256*8*8, out_features=512)
        self.lin2 = torch.nn.Linear(512, output)

    def forward(self, x):

        x = F.relu(self.cnn1(x))
        x = self.pool(F.relu(self.cnn2(x)))
        x = F.relu(self.cnn3(x))
        x = self.pool(F.relu(self.cnn4(x)))
        x = F.relu(self.cnn5(x))
        x = self.pool(F.relu(self.cnn6(x)))
        x = self.Bnorm(x)
        x = self.drop(x)
        x = x.view(-1, 256*8*8)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


#TripletNet -> Self-supervized architecture using previously pretrained network, Densnet
class TripNet(torch.nn.Module):
 
    def __init__(self, weights):

        super(TripNet, self).__init__()
        self.weights = weights
        self.denseNet = torch.load(weights, map_location=torch.device('cpu')) #my finetuned model
        self.denseNet.classifier = nn.Linear(self.denseNet.classifier.in_features, 1024)#.cuda()
        self.classifier = self.denseNet#.cuda()
        self.fc1 = torch.nn.Linear(1024, 2048)#.cuda()
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(2048, 1024)#.cuda()
        self.fc3 = torch.nn.Linear(1024, 512)#.cuda()

    
    def embeddings(self, x):
 
      embedding = self.classifier(x)
      embedding = self.relu(embedding)      
      embedding = self.fc1(embedding)
      embedding = self.relu(embedding)
      embedding = self.fc2(embedding)
      embedding = self.relu(embedding)
      embedding = self.fc3(embedding)
      
 
      return embedding
 
    def forward(self, x_a, x_p=None, x_n = None):
        
        a_embedings = self.embeddings(x_a)
        
        if x_p !=None and x_n != None:

          p_embedings = self.embeddings(x_p)
          n_embedings = self.embeddings(x_n)

           # L2 norm
          norm = a_embedings.norm(p=2, dim=1, keepdim=True)
          a_embedings = a_embedings.div(norm)
          norm = p_embedings.norm(p=2, dim=1, keepdim=True)
          p_embedings = p_embedings.div(norm)
          norm = n_embedings.norm(p=2, dim=1, keepdim=True)
          n_embedings = n_embedings.div(norm)

          # Check if triplet is already correct (not used for the loss, just for monitoring)
          correct = torch.zeros([1], dtype=torch.int32).cuda()
          dist_pos = F.pairwise_distance(a_embedings, p_embedings, p=2)
          dist_neg = F.pairwise_distance(a_embedings, n_embedings, p=2)

          for i in range(0,len(dist_pos)):
              if (dist_neg[i] - dist_pos[i]) >= 0.5:
                  correct[0] += 1

          return a_embedings, p_embedings, n_embedings, correct[0]
 
        
        # p_distance = torch.nn.functional.pairwise_distance(a_embedings, p_embedings)
        # n_distance = torch.nn.functional.pairwise_distance(a_embedings, n_embedings) #use this if going to use MarginRankingLoss

        norm = a_embedings.norm(p=2,dim=1,keepdim=True)
        a_embedings = a_embedings.div(norm)

        return a_embedings

#Transforming the self-supervised model into a supervised classifier by training a linear layer added at the bottom of the architecture
class TriNetClassifier(torch.nn.Module):

  def __init__(self, weights=None):

    super(TriNetClassifier, self).__init__()
    
    self.weights = weights
    self.featureExtractor = TripNet(self.weights) #CPU
    # self.featureExtractor = torch.load(self.weights) #GPU

    self.relu = torch.nn.ReLU(inplace=True)
    # self.fc3 = torch.nn.Linear(512, 2).cuda() #GPU
    self.fc3 = torch.nn.Linear(512, 2)


  def forward(self, x):

    x = self.featureExtractor(x)
    x = self.relu(x)
    x = self.fc3(x)

    return x