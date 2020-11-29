import torch
import torch.nn as nn
import torch.optim as optim
from data import Data
from model import Net

def train():

    # Getting the train and test set for CIFAR-10
    data_cifar10 = Data()
    train,test = data_cifar10.dataloader()

    #Defining the model
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)

    print(net)

    # Defining the loss function  and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Defining the training loop
    epochs =2
    for t in range(epochs):
        print("Epoch {}".format(t+1))
        for i,data in enumerate(train,0) :
            inputs,labels = data[0].to(device),data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # saving the model
    torch.save(net.state_dict(),'./cifar.pth')





if __name__ == '__main__':
    train()


