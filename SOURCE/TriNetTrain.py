import os
import torchvision.utils
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc
from torch.utils.tensorboard import SummaryWriter
import math
import cv2
import utilities
import dataset
import models
import random

np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)

WEIGHTS__ = os.path.join(os.getcwd(), "Model_Weights/Best_model_27June.pt")
BATCH_SIZE__ = 64
ROOT_DIR__ = os.path.join(os.getcwd(),"../Dataset")
SPLIT_DIR__ = os.path.join(os.getcwd(),"../Data-split")
train_NonCOVID__ = os.path.join(SPLIT_DIR__, "NonCOVID/trainCT_NonCOVID.txt")
train_COVID__ = os.path.join(SPLIT_DIR__, "COVID/trainCT_COVID.txt")
val_COVID__ = os.path.join(SPLIT_DIR__, "COVID/valCT_COVID.txt")
val_NonCOVID__ = os.path.join(SPLIT_DIR__, "NonCOVID/valCT_NonCOVID.txt") 


normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                std=[0.33165374, 0.33165374, 0.33165374])

train_transformer, val_transformer = utilities.transformers(normalize)

trainsetCT__ = dataset.CT_Triplet(root_dir=ROOT_DIR__,
                       split_covid=train_COVID__,
                       split_nonCovid=train_NonCOVID__,
                       transform= train_transformer)

valsetCT__ = dataset.CT_Triplet(root_dir=ROOT_DIR__,
                     split_covid=val_COVID__,
                     split_nonCovid=val_NonCOVID__,
                     transform= val_transformer,
                    train=False)

train_loader = DataLoader(trainsetCT__, batch_size=BATCH_SIZE__, drop_last=False, shuffle=True)
val_loader = DataLoader(valsetCT__, batch_size=BATCH_SIZE__, drop_last=False, shuffle=False)


def train(model, train_loader, optimizer, criterion, num_epochs):
    
    for epoch in range(num_epochs):
        
        print("Starting epoch:", epoch)
        runing_loss = 0
        correct_outputs = 0
        
        for i, data in enumerate(train_loader):
            #get the inputs 
            image_a, image_p, image_n, labels_a, labels_p, labels_n = data
 
            image_a = image_a.cuda()
            image_p = image_p.cuda()
            image_n = image_n.cuda()
 
 
            #zero the parameter gradients
            optimizer.zero_grad()
 
            #forward + backward + optimize
            a_embedings, p_embedings, n_embedings, correct = model(image_a, image_p, image_n)
            a_embedings = a_embedings.cuda()
            p_embedings = p_embedings.cuda()
            n_embedings = n_embedings.cuda()
 
            loss = criterion(a_embedings, p_embedings, n_embedings) #tripletloss(anchor, positive, negative) calculates similarities
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()  
            correct_outputs += correct.cpu()
        
       
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, runing_loss / (trainsetCT__.__len__()/BATCH_SIZE__)))
        print('[%d, %5d] Correct: %d, Num sampes: %d ' % (epoch + 1, i + 1, correct_outputs , trainsetCT__.__len__()))

        validate(model, val_loader, criterion, epoch, optimizer)
        
    return model

def validate(model, val_loader, criterion, epoch,optimizer):
        
    val_loss=0
    correct_outputs = 0
    
    with torch.no_grad():
        
        for data in val_loader:
            
            image_a, image_p, image_n, labels_a, labels_p, labels_n = data
            image_a = image_a.cuda()
            image_p = image_p.cuda()
            image_n = image_n.cuda()
 
            a_embedings, p_embedings, n_embedings, correct = model(image_a, image_p,image_n) 
            a_embedings = a_embedings.cuda()
            p_embedings = p_embedings.cuda()
            n_embedings = n_embedings.cuda()
 
            loss_val = criterion(a_embedings, p_embedings, n_embedings)
            val_loss +=loss_val.item()
            correct_outputs += correct.cpu()
  
        
        print('[%d] loss: %.3f' %(epoch + 1,  val_loss / (valsetCT__.__len__()/BATCH_SIZE__)))
        print('Correct: %d, Num sampes: %d ' % ( correct_outputs , valsetCT__.__len__()))
        

model = models.TripNet(WEIGHTS__)

params_to_update = []
for name,param in model.named_parameters():
        if  'classifier' in name or 'fc1' in name or "fc2" in name or "fc3" in name or 'denseblock4'in name:# or 'denseblock3' in name:
          if param.requires_grad == False:
              param.requires_grad = True
              params_to_update.append(param)
        
          else:
              params_to_update.append(param)

tripletCriterion = torch.nn.TripletMarginLoss(margin=0.5)
optimizer_ = torch.optim.SGD(params_to_update, lr=0.0005, momentum=0.9)
train(model=model,optimizer=optimizer_, criterion=tripletCriterion,train_loader=train_loader, num_epochs=300)



