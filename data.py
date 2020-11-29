import torch
import torchvision
import torchvision.transforms as transforms


class Data:
    def dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='/home/aer4kor/PycharmProjects/PP/cifar10_classification_pytorch/train', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/home/aer4kor/PycharmProjects/PP/cifar10_classification_pytorch/test', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)
        return trainloader,testloader



