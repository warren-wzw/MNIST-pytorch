#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import torch.optim as optim
from   torchvision import datasets , transforms
from   torch.utils.data import  DataLoader
#import cv2
import numpy as np
from simple_net import SimpleModel

#hyperparameter
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE     = torch.device("cuda")
EPOCH      = 5



#define training ways
def train_model(model,device,train_loader,optimizer,epoch):
    #training
    model.train()
    for batch_index,(data,target) in enumerate(train_loader):
        #deploy to  device
        data,target =data.to(device),target.to(device)
        #init gradient
        optimizer.zero_grad()
        #training results
        output = model(data)
        #calulate loss
        loss = F.cross_entropy(output,target)
        #find the best score's index
        #pred = output.max(1,keepdim = True)
        #backword
        loss.backward()
        optimizer.step()
        if batch_index % 3000 ==0:
            print("Train Epoch :{} \t Loss :{:.6f}".format(epoch,loss.item()))

#test
def test_model(model,device,test_loader):
    model.eval()
    #correct rate
    correct = 0.0
    #test loss
    test_loss=0
    with torch.no_grad(): #do not caculate gradient as well as backward
        for data,target in test_loader:
            datra,target = data.to(device),target.to(device)
            #test data
            output = model(data.to(device))
            #caculte loss
            test_loss += F.cross_entropy(output,target).item()
            #find the index of the best score
            pred =output.max(1,keepdim=True)[1]
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /=len(test_loader.dataset)
        print("TEST - average loss : {: .4f}, Accuracy :{:.3f}\n".format(
            test_loss,100.0*correct /len(test_loader.dataset)))

def main():
    #pipeline
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    #download dataset
    train_set    = datasets.MNIST("data",train=True,download=False,transform=pipeline)
    test_set     = datasets.MNIST("data",train=False,download=False,transform=pipeline)
    #load dataset
    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader  = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

    #show dataset
    with open("./data/MNIST/raw/train-images-idx3-ubyte","rb") as f:
        file =f.read()
    image1 = [int(str(item).encode('ascii'),16) for item in file[16:16+784]]
    #print(image1)
    image1_np=np.array(image1,dtype=np.uint8).reshape(28,28,1)
    #print(image1_np.shape)
    #cv2.imwrite("test.jpg",image1_np)

    #optim
    model     = SimpleModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    #9 recall function to train
    for epoch in range(1,EPOCH+1):
        train_model(model,DEVICE,train_loader,optimizer,epoch)
        test_model(model,DEVICE,test_loader)
    # Create a SimpleModel and save its weight in the current directory
    model_wzw = SimpleModel() 
    torch.save(model.state_dict(), "weight.pth")

if __name__ == "__main__":
    main()



