import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funcs
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.optim as opt
import time


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=40,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ResBlock_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        super(ResBlock_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
        )

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        F = self.conv(x)
        res = self.shortcut(x)
        output = F + res
        output = Funcs.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.a1 = ResBlock_2(in_channel=16, out_channel=16, kernel_size=3, stride=1, padding=1)
        self.a2 = ResBlock_2(in_channel=16, out_channel=16, kernel_size=3, stride=1, padding=1)
        self.a3 = ResBlock_2(in_channel=16, out_channel=16, kernel_size=3, stride=1, padding=1)

        self.b1 = ResBlock_2(in_channel=16, out_channel=32, kernel_size=3, stride=2, padding=1)
        self.b2 = ResBlock_2(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1)
        self.b3 = ResBlock_2(in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1)

        self.c1 = ResBlock_2(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1)
        self.c2 = ResBlock_2(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)
        self.c3 = ResBlock_2(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        output = self.conv_1(x)
        output = self.a1(output)
        output = self.a2(output)
        output = self.a3(output)
        output = self.b1(output)
        output = self.b2(output)
        output = self.b3(output)
        output = self.c1(output)
        output = self.c2(output)
        output = self.c3(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    net = ResNet()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    optimizer = opt.SGD(net.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)

    net.train()
    start_time = time.time()
    last_time = 0
    iteration = 0
    break_flag = False
    for epoch in range(82):

        if break_flag:
            break

        total = 0
        correct = 0

        for i, data in enumerate(trainloader, 0):
            iteration += 1
            if iteration == 32000:
                break_flag = True
                break
            if iteration == 16000 or iteration == 24000:
                lr = lr / 10
                optimizer = opt.SGD(net.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
                print("learning rate change to %f" % lr)

            inputs, labels = data
            inputs_V, labels_V = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs_V)
            loss = criterion(outputs, labels_V)
            loss.backward()
            optimizer.step()

            # print accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
            if i % 100 == 99:
                print('[%d] accuracy: %d %%' % (iteration, 100 * correct / total))
                total = 0
                correct = 0
                now_time = time.time()
                print("It took %ds" % (now_time - last_time))
                last_time = now_time

    print('Finished Training. It took %ds in total' % (time.time() - start_time))

    # test
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = net(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))




































