#https://blog.csdn.net/briblue/article/details/100693365
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
         transforms.Normalize((0.1307,), (0.3081,))
         ])

transform_test = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
         ])

trainset = datasets.FashionMNIST(root='./fashionmnist_data', train=True,
                                            download=False, transform=transform_train)
train_loader = utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

testset = datasets.FashionMNIST(root='./fashionmnist_data', train=False,
                                           download=False, transform=transform_test)
test_loader = utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

class Net(nn.Module):


    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,1,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        self.fc6 = nn.Linear(512,10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        #print(" x shape ",x.size())
        x = x.view(-1,128*8*8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return x

    def train_sgd(self,device,epochs=100, modelname='model.tar'):

        #cnn = CNN(); cnn = cnn.to(device)
        LEARNING_RATE = 0.0001
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        #optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        
        path = modelname
        initepoch = 0

        if os.path.exists(path) is not True:
            loss_criterion = nn.CrossEntropyLoss()
            loss_list = []
            #optimizer = optim.SGD(self.parameters(),lr=LEARNING_RATE)

        else:
            loss_list = []
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss_criterion = checkpoint['loss_criterion']

        f = open("out_custom_%s.txt" % modelname.split('.')[0], 'w')
        for epoch in range(initepoch, epochs):
          # loop over the dataset multiple times
            timestart = time.time()
            running_loss = 0.0; total = 0; correct = 0;

            for i, (images, labels) in enumerate(train_loader, 0):
                # get the inputs
                images = images.to(device)
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

    def test(self, device, modelname):
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
    optim_type = 'vggNN_' + optim_type + '.tar'
    epochs = int(input("Specify times of iterations [epochs]: "))
    net = Net()
    net = net.to(device)
    net.train_sgd(device, epochs, optim_type)
    net.test(device, optim_type)


