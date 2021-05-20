# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:53:45 2019

@author: imbroscb
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
from im_converter import im_convert
from vesicle_classifier import MultiClass
from dataset_modifications import AddGaussianNoise
#%%
torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True 

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

#%% transform with data  and load data
transform_train=transforms.Compose([transforms.Resize((40,40)),
                                    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                    transforms.RandomApply([AddGaussianNoise(0,0.05)],p=0.1),
                                    transforms.RandomApply([AddGaussianNoise(0,0.1)],p=0.2)]) 



transform_test=transforms.Compose([transforms.Resize((40,40)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 

training_dataset=datasets.ImageFolder(root=os.path.join('data','train'),transform=transform_train) #training and validation dataset is already divided
training_loader=torch.utils.data.DataLoader(dataset=training_dataset,batch_size=100, shuffle=True)

testing_dataset=datasets.ImageFolder(root=os.path.join('data','test'),transform=transform_test) #training and validation dataset is already divided

validation_loader=torch.utils.data.DataLoader(dataset=testing_dataset,batch_size=100, shuffle=False)

#%%
#check dimentions

print('Batch size: ',training_loader.batch_size)
print('Length training dataset: ',len(training_dataset))
print('Length validation dataset: ',len(testing_dataset))
dataiter=iter(training_loader) # get one image + the corresponding label at a time calling next
images,labels=dataiter.next()
print('Image shape: ',images.shape)

#%% Defining the model

model=MultiClass(out=2).to(device)
criterion=nn.CrossEntropyLoss() 
optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)

#%% Training

running_loss_history=[]
running_correct_history=[]
running_precision_history=[]
running_recall_history=[]
val_running_loss_history=[]
val_running_correct_history=[]
val_running_precision_history=[]
val_running_recall_history=[]
counter_list=[]
#%%
epochs=25

for e in range(epochs):
    running_loss=0.0
    running_correct=0.0
    val_running_loss=0.0
    val_running_correct=0.0
    val_counter=0
    counter=0
    val_counter=0

    true_pos=0
    val_true_pos=0
    prec_denom =0
    val_prec_denom=0
    rec_denom=0
    val_rec_denom=0
    for inputs,labels in training_loader: 
        counter+=inputs.shape[0]
        inputs = inputs.to(device)
        labels=labels.to(device)    
        inputs=inputs[:,0,:,:]
        inputs=inputs.view(inputs.shape[0],1,inputs.shape[1],inputs.shape[2])
        output=model.forward(inputs) 
        loss=criterion(output,labels) 

        optimizer.zero_grad() # set gradients to zero
        loss.backward() # calculate the gradients 
        optimizer.step() # update the weight according to the gradient and the lr

        running_loss += loss.item() 

        _,preds = torch.max(output,1) 
        running_correct+= torch.sum(preds == labels.data)  

        for i in range(preds.shape[0]):
            if preds[i]==1:
                if preds[i]==labels[i].data:
                    true_pos+=1
                    
        prec_denom+= torch.sum(preds.cpu() == 1.0) # remove .cpu() if training was done on cpu
        rec_denom+=torch.sum(labels.data.cpu()==1.0) # remove .cpu() if training was done on cpu
        
    else:     

        with torch.no_grad():
            for val_inputs,val_labels in validation_loader:
                val_inputs=val_inputs.to(device)
                val_labels=val_labels.to(device)
                val_counter+=val_inputs.shape[0]
                val_inputs=val_inputs[:,0,:,:]
                val_inputs=val_inputs.view(val_inputs.shape[0],1,val_inputs.shape[1],val_inputs.shape[2])

                val_output=model.forward(val_inputs)
                val_loss=criterion(val_output,val_labels)

                val_running_loss += val_loss.item()

                _,val_preds=torch.max(val_output,1)
                val_running_correct += torch.sum(val_preds==val_labels.data)

                for i in range(val_preds.shape[0]):
                    if val_preds[i]==1:
                        if val_preds[i]==val_labels[i].data:
                            val_true_pos+=1
                            
                val_prec_denom+= torch.sum(val_preds.cpu() == 1.0) # remove .cpu() if training was done on cpu
                val_rec_denom+=torch.sum(val_labels.data.cpu()==1.0) # remove .cpu() if training was done on cpu

    epoch_loss=running_loss/counter*100
    epoch_accuracy=running_correct/counter*100
    true_pos=np.array(true_pos)
    
    val_epoch_loss=val_running_loss/val_counter*100
    val_epoch_accuracy=val_running_correct/val_counter*100
    val_true_pos=np.array(val_true_pos)
    
    prec_denom=prec_denom.numpy()
    rec_denom=rec_denom.numpy()    
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_accuracy)
    running_precision_history.append(true_pos/prec_denom*100)    
    running_recall_history.append(true_pos/rec_denom*100)   
            
    val_prec_denom=val_prec_denom.numpy()
    val_rec_denom=val_rec_denom.numpy()    
    val_running_loss_history.append(val_epoch_loss)
    val_running_correct_history.append(val_epoch_accuracy)
    val_running_precision_history.append(val_true_pos/val_prec_denom*100)    
    val_running_recall_history.append(val_true_pos/val_rec_denom*100)    

    print('Epoch, training loss, training accuracy: {:},{:.4f},{:.4f}'.format(e,epoch_loss,epoch_accuracy)) #,{:.d} this is called place holder
    print('Epoch, validation loss, validation accuracy: {:},{:.4f},{:.4f}'.format(e,val_epoch_loss,val_epoch_accuracy)) #,{:.d} this is called place holder
    print('Epoch, training precision, training recall: {:},{:.4f},{:.4f}'.format(e,running_precision_history[-1],running_recall_history[-1])) #,{:.d} this is called place holder
    print('Epoch, validation precision, validation recall: {:},{:.4f},{:.4f}'.format(e,val_running_precision_history[-1],val_running_recall_history[-1])) #,{:.d} this is called place holder
    print('---------------------------------------------------------------------------')
    
    # here change accordingly the directory where to save the weights of the model after each epoch
    torch.save(model.state_dict(), os.path.join('/path','model_dir', 'epoch-{}.pth'.format(e)))

    
#%%
f1=[]
for i in range(epochs):
    f1.append(2*(val_running_precision_history[i]*val_running_recall_history[i])/(val_running_precision_history[i]+val_running_recall_history[i]))    

f1_train=[]
for i in range(epochs):
    f1_train.append(2*(running_precision_history[i]*running_recall_history[i])/(running_precision_history[i]+running_recall_history[i]))    

      



