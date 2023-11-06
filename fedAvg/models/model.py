import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math

# Define Logistic model for MNIST and CIFAR10
class Logistic(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits

# Define 2nn model for MNIST and CIFAR10
class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_size=28*28, hidden_size1=100, hidden_size2=50, num_classes=10):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden1 = self.fc1(x)
        hidden2 = self.fc2(hidden1)
        logits = self.fc3(hidden2)
        return logits

# Define the LeNet-5 model for MNIST
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the CNN model for MNIST (as RoFL, params number: 19166)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        # self.conv2 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        # self.pool = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(32 * 7 * 7, 256)
        # self.fc2 = nn.Linear(256, num_classes)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)
        # print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the LeNet-5 model for CIFAR10 (as RoFL, params number: 62006)
class LeNet5C(nn.Module):
    def __init__(self):
        super(LeNet5C, self).__init__()
        input_shape, num_classes, do = (3, 32, 32), 10, 0
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(do)
        # print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Define the LeNet-20 model for CIFAR10
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class LeNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet20, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 20, 2, stride=1)
        self.layer2 = self._make_layer(20, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def choose_model(options):
    model_name = str(options['model']).lower()
    dataset_name = options['dataset'].split('_')[0]

    if dataset_name == 'mnist':
        if model_name == 'logistic':
            return Logistic(input_size=28*28)
        elif model_name == '2nn':
            return TwoHiddenLayerFc(input_size=28*28)
        elif model_name == 'lenet':
            return LeNet5()
        elif model_name == 'cnn':
            return CNN()

    elif dataset_name == 'cifar10':
        if model_name == 'logistic':
            return Logistic(input_size=3*32*32)
        elif model_name == '2nn':
            return TwoHiddenLayerFc(input_size=3*32*32)
        elif model_name == 'lenet':
            return LeNet20()
        elif model_name == 'lenet5':
            return LeNet5C()


    else:
        raise ValueError("Not support model[{}] of dataset[{}]!".format(model_name, dataset_name))
