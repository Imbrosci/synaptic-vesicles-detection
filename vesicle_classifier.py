# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:08:53 2019

@author: imbroscb
"""
#%%
import torch
from torch import nn
import torch.nn.functional as F


class MultiClass(nn.Module):     
                 
    def __init__(self,conv_out1=16,conv_out2=32,conv_out3=64,conv_out4=128,filtersize=7,strd=1,paddingsize=2,H1=300,H2=150,H3=50,out=2): #size of input layer, first and second hidden layer and output layer
        super().__init__()
        self.conv_out1=conv_out1
        self.conv_out2=conv_out2
        self.conv_out3=conv_out3
        self.conv_out4=conv_out4

        self.filtersize=filtersize
        self.strd=strd

        self.H1=H1
        self.H2=H2
        self.H3=H3
        
        self.out=out
        self.paddingsize=paddingsize
        
        cutting=self.filtersize-1-(2*self.paddingsize) 
        
        #calculating feature map dimension after convolution-pooling layers
        dimension=40
        for conv in range(4):
            dimension=dimension-cutting
            if dimension % 2 ==1:
                dimension-=1
            dimension=dimension/self.strd
        self.input1=int(dimension/2)
        
        # defining convolutional layers
        self.conv1=nn.Conv2d(1,self.conv_out1,self.filtersize,self.strd,padding=self.paddingsize) 
        self.conv2=nn.Conv2d(self.conv_out1,self.conv_out2,self.filtersize,self.strd,self.paddingsize)
        self.conv3=nn.Conv2d(self.conv_out2,self.conv_out3,self.filtersize,self.strd,self.paddingsize)
        self.dropout_c34=nn.Dropout(0.3) 
        self.conv4=nn.Conv2d(self.conv_out3,self.conv_out4,self.filtersize,self.strd,self.paddingsize)        
        self.dropout_c4f1=nn.Dropout(0.3) 

        # defining fully connected layers        
        self.fc1=nn.Linear(self.input1*self.input1*self.conv_out4,self.H1)
        self.dropout_f12=nn.Dropout(0.75) 
        self.fc2=nn.Linear(self.H1,self.H2)
        self.dropout_f23=nn.Dropout(0.65) 
        self.fc3=nn.Linear(self.H2,self.H3)
        self.dropout_f34=nn.Dropout(0.55) 
        self.fc4=nn.Linear(self.H3,self.out)

        
    def forward(self,x):
        
        # applying the convolution operation and the activation function ReLU
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))         
        x=F.relu(self.conv3(x)) 
        x=self.dropout_c34(x) 
        x=F.relu(self.conv4(x)) 
        
        # max pooling operation 
        x=F.max_pool2d(x,2,2) 
        x=self.dropout_c4f1(x) 
 
        # fattening x
        x=x.view(-1,self.input1*self.input1*self.conv_out4)

        # appling ReLU on the output from fully connected layers
        x= F.relu(self.fc1(x)) 
        x=self.dropout_f12(x)        
        x=F.relu(self.fc2(x))
        x=self.dropout_f23(x)        
        x=F.relu(self.fc3(x))
        x=self.dropout_f34(x)        
        
        x=self.fc4(x)

        return x  
#%%
class MultiClassPost(nn.Module):     
                 
    def __init__(self,conv_out1=16,conv_out2=32,conv_out3=64,conv_out4=128,filtersize=7,strd=1,paddingsize=2,H1=300,H2=150,H3=50,out=2): #size of input layer, first and second hidden layer and output layer
        super().__init__()
        self.conv_out1=conv_out1
        self.conv_out2=conv_out2
        self.conv_out3=conv_out3
        self.conv_out4=conv_out4

        self.filtersize=filtersize
        self.strd=strd

        self.H1=H1
        self.H2=H2
        self.H3=H3
        
        self.out=out
        self.paddingsize=paddingsize
        
        cutting=self.filtersize-1-(2*self.paddingsize) 
        
        #calculating feature map dimension after convolution-pooling layers
        dimension=80
        for conv in range(4):
            dimension=dimension-cutting
            if dimension % 2 ==1:
                dimension-=1
            dimension=dimension/self.strd
        self.input1=int(dimension/2)
        
        # defining convolutional layers
        self.conv1=nn.Conv2d(1,self.conv_out1,self.filtersize,self.strd,padding=self.paddingsize) 
        self.conv2=nn.Conv2d(self.conv_out1,self.conv_out2,self.filtersize,self.strd,self.paddingsize)
        self.conv3=nn.Conv2d(self.conv_out2,self.conv_out3,self.filtersize,self.strd,self.paddingsize)
        self.dropout_c34=nn.Dropout(0.3) 
        self.conv4=nn.Conv2d(self.conv_out3,self.conv_out4,self.filtersize,self.strd,self.paddingsize)        
        self.dropout_c4f1=nn.Dropout(0.3) 

        # defining fully connected layers        
        self.fc1=nn.Linear(self.input1*self.input1*self.conv_out4,self.H1)
        self.dropout_f12=nn.Dropout(0.75) 
        self.fc2=nn.Linear(self.H1,self.H2)
        self.dropout_f23=nn.Dropout(0.65) 
        self.fc3=nn.Linear(self.H2,self.H3)
        self.dropout_f34=nn.Dropout(0.55) 
        self.fc4=nn.Linear(self.H3,self.out)

        
    def forward(self,x):
        
        # applying the convolution operation and the activation function ReLU
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))         
        x=F.relu(self.conv3(x)) 
        x=self.dropout_c34(x) 
        x=F.relu(self.conv4(x)) 
        
        # max pooling operation 
        x=F.max_pool2d(x,2,2) 
        x=self.dropout_c4f1(x) 
 
        # fattening x
        x=x.view(-1,self.input1*self.input1*self.conv_out4)

        # appling ReLU on the output from fully connected layers
        x= F.relu(self.fc1(x)) 
        x=self.dropout_f12(x)        
        x=F.relu(self.fc2(x))
        x=self.dropout_f23(x)        
        x=F.relu(self.fc3(x))
        x=self.dropout_f34(x)        
        
        x=self.fc4(x)

        return x  