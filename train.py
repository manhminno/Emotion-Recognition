import cv2
import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
import numpy as np
import time


from model import *

#Load data
train_set = datasets.ImageFolder(root='./data/train', transform=transform_train)
val_set = datasets.ImageFolder(root='./data/val', transform=transform_val)

batch_size = 32

train_load = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_load = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

train_loss = []
val_loss = []
train_acc = []
val_acc = []

#Def training model 
def Training_Model(model, epochs, parameters):
    #Using CrossEntropyLoss, optim SGD
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters, lr=0.07, weight_decay=0.00001)

    model = model.cuda()
    
    for epoch in range(epochs): 
        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train() #Set mode Train                  
        
        for i, (inputs, labels) in enumerate(train_load, 0):
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad() 
            
            outputs = model(inputs)    
            loss = loss_f(outputs, labels)
            iter_loss += loss.item()
            
            loss.backward()              
            optimizer.step()             
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1
    

        train_loss.append(iter_loss/iterations)
        train_acc.append((100 * correct / len(train_set)))
   
        #val_eval
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval() #Set mode evaluation

        #No_grad on Val_set
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_load, 0):
                
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                outputs = model(inputs)     
                loss = loss_f(outputs, labels) 
                loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            val_loss.append(loss/iterations)
            val_acc.append((100 * correct / len(val_set)))

        stop = time.time()
        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}, Time: {}s'
            .format(epoch+1, epochs, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1],stop-start))

# Training model
model = CNN()
model = model.cuda()
# model.load_state_dict(torch.load('Emotion-Detection.pth'))

epochs = 32
Training_Model(model=model, epochs=epochs, parameters=model.parameters())
# torch.save(model.state_dict(),'Emotion-Detection.pth')