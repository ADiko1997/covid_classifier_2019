import random
import os
import shutil

home=os.path.getcwd()
N_patient = os.listdir(home + "ct_lesion_seg/mask")          #creo lista con file dentro mask: cartelle con numero paziente
del N_patient[0]                                      #rimuovo Ds_store dala lista
random.shuffle(N_patient)                             #dispone in modo casuale i pazienti nella lista N_patient
  
    
newpath = home + 'test'
if not os.path.exists(newpath):                     #crea la cartella test
    os.makedirs(newpath)

    
newpath = home + 'train'
if not os.path.exists(newpath):                     #crea la cartella train
    os.makedirs(newpath)

    
base_dir=home + 'ct_lesion_seg\mask'                   #definisco due variabili base_dir che mi serviranno per concatenare path
base_dir1=home + 'ct_lesion_seg\image'


for i in range(0,120):
    N_slice=os.listdir(home + "ct_lesion_seg\mask\\"+N_patient[i])             #creo lista con i file dentro la cartella mask
    #N_slice=os.listdir(home+"ct_lesion_seg\mask"+N_patient[i])              prova su linux
    for j in range(0,5):
        Name=N_patient[i]+'_'+N_slice[j]
        
        in_name=os.path.join(base_dir,N_patient[i],N_slice[j])
        out_name=os.path.join(base_dir,N_patient[i],Name)              #rinomino i file dentro mask 
        os.rename(in_name,out_name)
        
        in_name1=os.path.join(base_dir1,N_patient[i],N_slice[j][:-4]+'.jpg')
        out_name1=os.path.join(base_dir1,N_patient[i],Name[:-4]+'.jpg')           #rinomino i file dentro image
        os.rename(in_name1,out_name1)
        
        shutil.move(out_name,home+'train')                                   #sposto i file da image e mask a train
        shutil.move(out_name1,home+'train')
        
        
for i in range(120,150):
    N_slice=os.listdir(home + "ct_lesion_seg\mask\\" + N_patient[i])             #Stesso algo visto prima. Metto mask e image degli ultimi 30 pazienti della lista N_patient, disposti precedentemente in ordine casuale, in test
    #N_slice=os.listdir(home + "ct_lesion_seg\mask" + N_patient[i])            prova su linux
    for j in range(0,5):
        Name=N_patient[i]+'_'+N_slice[j]
        
        in_name=os.path.join(base_dir,N_patient[i],N_slice[j])
        out_name=os.path.join(base_dir,N_patient[i],Name)
        os.rename(in_name,out_name)
        
        in_name1=os.path.join(base_dir1,N_patient[i],N_slice[j][:-4]+'.jpg')
        out_name1=os.path.join(base_dir1,N_patient[i],Name[:-4]+'.jpg')
        os.rename(in_name1,out_name1)
        
        shutil.move(out_name,home + 'test') 
        shutil.move(out_name1,home + 'test')
        