#https://zhuanlan.zhihu.com/p/96242688
import torch,math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import time
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils as utils

BATCH_SIZE = 128 
TEST_BATCH_SIZE = 50

transform_train = transforms.Compose(
        [
         transforms.RandomHorizontalFlip(),
         transforms.RandomGrayscale(),
         transforms.ToTensor(),
         #transforms.Normalize((0.1307,), (0.3081,))
         ])

transform_test = transforms.Compose(
        [
         transforms.ToTensor(),
         #transforms.Normalize((0.1307,), (0.3081,))
         ])

trainset = datasets.FashionMNIST(root='./fashionmnist_data', train=True,
                                            download=False, transform=transform_train)
train_loader = utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

testset = datasets.FashionMNIST(root='./fashionmnist_data', train=False,
                                           download=False, transform=transform_test)
test_loader = utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, padding =2),
            nn.BatchNorm2d(16),
            nn.ReLU()) #16, 28, 28
        self.pool1 = nn.MaxPool2d(2) #16, 14, 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU()) #32, 12, 12
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU()) #64, 10, 10
        self.pool2 = nn.MaxPool2d(2) #64, 5, 5
        self.fc = nn.Linear(5*5*64, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out=self.pool1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out=self.layer3(out)
        #print(out.shape)
        out=self.pool2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out 

    def train_adam(self,device,epochs=100, modelname='model.tar'):

        #cnn = CNN(); cnn = cnn.to(device)
        #LEARNING_RATE = 0.01
        LEARNING_RATE = 0.0001
        #optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE)
        optimizer = optim.SGD(self.parameters(), lr = 0.01, momentum=0.9)

        path = modelname
        initepoch = 0

        if os.path.exists(path) is not True:
            loss_criterion = nn.CrossEntropyLoss()
            loss_list = []

        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss_criterion = checkpoint['loss_criterion']
            loss_list = []


        f = open("out_custom_%s.txt" % modelname.split('.')[0], 'w')
        for epoch in range(initepoch, epochs):

            timestart = time.time()
            running_loss = 0.0; total = 0; correct = 0;

            for i, (images, labels) in enumerate(train_loader):
                images = images.float().to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad() 

                # forward + backward + optimize
                outputs = self(images)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                #loss_list.append(loss.cpu().data.item())
                running_loss += loss.item()

                if i % 500 == 499:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i, running_loss / 500), file=f)
                    loss_list.append(loss.cpu().data.item())
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on %d training images: %.3f %%' % (total,
                            100.0 * correct / total), file=f)
                    total = 0
                    correct = 0
                    torch.save({'epoch':epoch, 'model_state_dict':net.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'loss_criterion':loss_criterion},path)

            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart), file=f)

        print('Finished Training', file=f)
        f.close()
        #plt.xkcd(); 
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.plot(loss_list); plt.show(); plt.savefig("custom_%s.jpg" % modelname.split('.')[0])

    def test(self,device, modelname):
        correct = 0; total = 0
        y_true = [], y_pred = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.append(labels); y_pred.append(predicted)
                print(y_pred); print(y_true)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        with open("out_custom_%s.txt" % modelname.split('.')[0], 'a') as f:
            print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total), file=f)
        y_true = list(np.array(y_true).flatten());
        y_pred = list(np.array(y_pred).flatten());
        from analysis import my_classification_report
        analysis.my_classification_report(y_true, y_pred, True)




if __name__ == "__main__":

    classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optim_type = input("Specify types of optimizer [adam/sgd]: ")
    optim_type = 'cNN_' + optim_type + '.tar'
    epochs = int(input("Specify times of iterations [epochs]: "))
    net = CNN()
    net = net.to(device)
    net.train_adam(device, epochs, optim_type)
    net.test(device, optim_type)

