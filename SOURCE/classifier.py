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
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m", help="set output width")
test = parser.parse_args()

np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)

WEIGHTS__ = os.path.join(os.getcwd(), "Model_Weights/final.pt")
BATCH_SIZE__ = 128
ROOT_DIR__ = os.path.join(os.getcwd(),"../Dataset")
SPLIT_DIR__ = os.path.join(os.getcwd(),"../Data-split")

train_NonCOVID__ = os.path.join(SPLIT_DIR__, "NonCOVID/trainCT_NonCOVID.txt")
train_COVID__ = os.path.join(SPLIT_DIR__, "COVID/trainCT_COVID.txt")
val_COVID__ = os.path.join(SPLIT_DIR__, "COVID/valCT_COVID.txt")
val_NonCOVID__ = os.path.join(SPLIT_DIR__, "NonCOVID/valCT_NonCOVID.txt") 
test_NonCOVID__ = os.path.join(SPLIT_DIR__, 'NonCOVID/testCT_NonCOVID.txt')
test_COVID__ = os.path.join(SPLIT_DIR__, 'COVID/testCT_COVID.txt')


normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

train_transformer, val_transformer = utilities.transformers(normalize)

trainsetCT__ = dataset.CT_Images(root_dir=ROOT_DIR__,
                       split_covid=train_COVID__,
                       split_nonCovid=train_NonCOVID__,
                       transform= train_transformer)

valsetCT__ = dataset.CT_Images(root_dir=ROOT_DIR__,
                     split_covid=val_COVID__,
                     split_nonCovid=val_NonCOVID__,
                     transform= val_transformer)

testsetCT__ = dataset.CT_Images(root_dir=ROOT_DIR__,
                      split_covid=test_COVID__,
                      split_nonCovid=test_NonCOVID__,
                      transform= val_transformer)

train_loader = DataLoader(trainsetCT__, batch_size=BATCH_SIZE__, drop_last=False, shuffle=True)
val_loader = DataLoader(valsetCT__, batch_size=BATCH_SIZE__, drop_last=False, shuffle=False)
test_loader = DataLoader(testsetCT__, batch_size=BATCH_SIZE__, drop_last=False, shuffle=False)

def train(model, train_loader, optimizer, criterion, num_epochs):

    best_model_loss = 1 
    test_acc = 0 
    for epoch in range(num_epochs):

        print("Starting epoch:", epoch)
        runing_loss = 0
        pred = []
        labels_list = []
        
        for i, data in enumerate(train_loader):
            
            #get the inputs 
            inputs, labels = data
            #set you variables to cuda, can be done even earlier so the values goes directly to GPU

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            output = model(inputs.cuda())
            loss = criterion(output.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()

            runing_loss += loss.item()  
            output = output.cpu()      
            pred.append(output.argmax(dim=1, keepdim=True).reshape(1,-1).detach().numpy()[0])
            labels_list.append(labels.numpy())


                

            if (math.ceil(trainsetCT__.__len__()/BATCH_SIZE__) / (i+1)) == 1: #print statistics at the end of each epoch

                print("Accuracy:", accuracy_score(np.concatenate(labels_list, axis=0), np.concatenate(pred, axis=0)) )
                print("F1-score:", f1_score(np.concatenate(labels_list, axis=0), np.concatenate(pred, axis=0)))
                fpr, tpr, thresholds = roc_curve(np.concatenate(labels_list, axis=0), np.concatenate(pred, axis=0))
                auc_score = auc(fpr, tpr)
                print("AUC:",auc_score)
            
        
       
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, runing_loss / (trainsetCT__.__len__()/BATCH_SIZE__)))        
        validate(model, val_loader, criterion, epoch, optimizer, best_model_loss)

        
    return model


def validate(model, val_loader, criterion, epoch,optimizer, best_model_loss):
          
    val_loss=0
    val_pred = []
    val_labels_list = []
    
    with torch.no_grad():
        
        for data in val_loader:
            
            images, labels = data
            # images, labels = images.cuda(), labels.cuda() #copy to GPU
            output = model(images.cuda())
            
            loss_val = criterion(output.cuda(), labels.cuda())
            val_loss +=loss_val.item()
            output = output.cpu()
            val_pred.append(output.argmax(dim=1, keepdim=True).reshape(1,-1).detach().numpy()[0])
            val_labels_list.append(labels.numpy())
            
        #Validation statistics   
        print("Validatio Accuracy:", accuracy_score(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0)) )
        print("Validatio F1-score:", f1_score(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0)))
        
        fpr, tpr, thresholds = roc_curve(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0))
        val_auc_score = auc(fpr, tpr)
        
        print("Validatio AUC:",val_auc_score)
        print('[%d] loss: %.3f' %
        (epoch + 1,  val_loss / (valsetCT__.__len__()/BATCH_SIZE__))) 
       
        return 

def TEST(model, test_loader):
    
    model.eval()
    val_pred = []
    val_labels_list = []
    
    with torch.no_grad():
        
        for data in test_loader:
            
            images, labels = data
            output = model(images.cuda()) #GPU
            # output = model(images) #CPU
            val_pred.append(output.argmax(dim=1, keepdim=True).reshape(1,-1).detach().numpy()[0])
            val_labels_list.append(labels.numpy())
            
        #Validation statistics   
        print("Test Accuracy:", accuracy_score(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0)) )
        print("TEST F1-score:", f1_score(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0)))
        
        fpr, tpr, thresholds = roc_curve(np.concatenate(val_labels_list, axis=0), np.concatenate(val_pred, axis=0))
        val_auc_score = auc(fpr, tpr)
        
        print("TEST AUC:",val_auc_score)


if test.mode:

    print("TEST Started") 
    model = models.TriNetClassifier((os.path.join(os.getcwd(), "Model_Weights/Best_model_27June.pt")))
    state_dict = torch.load(os.path.join(os.getcwd(),'Model_Weights/final2.pt'),map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=state_dict)
    TEST(model, test_loader)
    print("TEST FINISHED")
    sys.exit(0)


model = models.TriNetClassifier((os.path.join(os.getcwd(), "Model_Weights/Best_model_27June.pt")))
print("Start Training")

params_to_update = []
for name,param in model.named_parameters():
        if  'classifier' in name or 'fc1' in name or "fc2" in name or 'fc3'in name or 'denseblock4' in name:
          if param.requires_grad == False:
              param.requires_grad = True
              params_to_update.append(param)
        
          else:
              params_to_update.append(param)

optimizer= torch.optim.SGD(params_to_update, lr=0.0003, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
criterion__ = torch.nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([0.81,1.0]))
# criterion__ = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.81,1.0]))

train(model=model,optimizer=optimizer, criterion=criterion__,train_loader=train_loader, num_epochs=220)



