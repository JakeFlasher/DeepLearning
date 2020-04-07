#https://zhuanlan.zhihu.com/p/96242688
import torch,math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch.nn as nn
    
BATCH_SIZE = 256

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        self.Y = np.array(data.iloc[:, 0])
        del data #??data??????,????
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)
    
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

def plot_loss(losses):
    plt.xkcd()
    plt.xlabel('Epoch#')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show(); plt.savefig('loss_custom_cNN.jpg')

def plot_eval(cnn):
    cnn.eval()
	correct = 0
	total = 0
	for images, labels in test_loader:
	    images = images.float().to(DEVICE)
	    outputs = cnn(images).cpu()
	    _, predicted = torch.max(outputs.data, 1) #??????
	    total += labels.size(0)
	    correct += (predicted == labels).sum()
	print('???: %.4f %%' % (100 * correct / total))
        
def train(train_loader, train_dataset, TOTAL_EPOCHS=50):

    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    print(DEVICE)
    cnn = CNN()
    cnn = cnn.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    LEARNING_RATE=0.01
    optimizer = torch.optim.Adam(cnn.parameters(), lr = LEARNING_RATE)

    losses = []
    for epoch in range(TOTAL_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.float().to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad() #????
            outputs = cnn(images)
            loss = criterion(outputs, labels)  #??????
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().data.item())
            if (i+1) % 100 == 0:
                print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE, loss.data.item()))        
    
    plot_loss(losses)

if __name__ == "__main__":

    use_csv = False
    #use_csv = True
    
    if (use_csv == True):
        DATA_PATH = Path('./data/')
        train = FashionMNISTDataset(csv_file=DATA_PATH / "fashion_train.csv")
        test = FashionMNISTDataset(csv_file=DATA_PATH / "fashion_test.csv")
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)
        a = iter(train_loader)
        data = next(a)
        img = data[0][0].reshape(28, 28)
        print(data[0][0].shape)
        print(img.shape)
        plt.imshow(img, cmap=plt.cm.gray); plt.show()
    
    else:
        transform_train = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        trainset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=True,
                                                download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root='./fashionmnist_data', train=False,
                                               download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=50,
                                                 shuffle=False, num_workers=2)
        
    #cnn =CNN()
    #cnn(torch.rand(1, 1, 28, 28))
    #print(cnn)

    train(train_loader, trainset)    

