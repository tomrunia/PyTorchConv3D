import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Define the weights
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Forward function has to be defined. The backward function is automatically
    # defined when using autograd.
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2,2))
        # Flatten tensor
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:] # remove batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":

    batch_size = 128

    net = Net()
    net = net.cuda()

    # Define input pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Train set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('train_set', len(trainset))
    print('test_set', len(testset))

    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=1e-2)

    for epoch in range(50):

        print("#"*60)
        print("Epoch {}".format(epoch))

        for step, (images, labels) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            # Forward pass through the network
            output = net(images)

            # Compute the loss for this batch
            loss = criterion(output, labels)
            loss.backward()

            # Perform the gradient update
            optimizer.step()

            print(step, epoch, loss.data.cpu().numpy())
