#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Build a simple nn model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,3)
        self.fc1   = nn.Linear(20*10*10,500)
        self.fc2   = nn.Linear(500,10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(input_size,-1)  
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x,dim=1)
        #print("------------------------------output is ",output)
        return output


""" # Build a simple nn model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.relu1 = nn.ReLU()           
        self.pool1 = nn.MaxPool2d(2)     
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.relu2 = nn.ReLU()           
        self.pool2 = nn.MaxPool2d(2)     
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        #input_size = x.size[0]     #batch size * 1*28*28
        y = self.conv1(x)          # input batch*1*28*28 output batch*6*24*10(24=28-5+1)
        y = self.relu1(y)          #keep size of input 
        y = self.pool1(y)          # batch*6*12*12(12=24/2)
        y = self.conv2(y)          # input batch*6*12*12 output batch*16*8*8(24=28-5+1)
        y = self.relu2(y)          #keep size of input
        y = self.pool2(y)          # batch*16*4*4(6=12/2)
        y = y.view(y.shape[0], -1) #拉伸 接起来 -1：自动计算维度 256
        y = self.fc1(y)            #batch*256 output batch *120
        y = self.relu3(y)          #keep size of input
        y = self.fc2(y)            #batch*120 output*84
        y = self.relu4(y)          #keep size of input
        y = self.fc3(y)            #batch*84 output*10
        y = self.relu5(y)
        output = F.log_softmax(y,dim=1) #calculate score for each
        #print("------------------------------output is ",output)
        return output """
 