import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# # specify loss function (categorical cross-entropy)
# criterion = nn.CrossEntropyLoss()

# # specify optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# # define the CNN architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 48x48x1 image tensor)
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)   #gray scale image
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)   #gray scale image 
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        # convolutional layer (sees 24x24x64 tensor)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        # convolutional layer (sees 12x12x128 tensor)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        # convolutional layer (sees 6x6x256 tensor)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        # self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(256, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(256, 512, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (512 * 3 * 3 -> 1152)
        self.fc1 = nn.Linear(512 * 3 * 3, 1152)
        self.dense1_bn = nn.BatchNorm1d(1152)
        # linear layer (1152 -> 576)
        self.fc2 = nn.Linear(1152, 576)
        self.dense2_bn = nn.BatchNorm1d(576)
        # linear layer (576 -> 7)
        self.fc3 = nn.Linear(576, 7)  #7 different output
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = (F.relu(self.conv1_1_bn(self.conv1_1(x))))  #48x48 16 foto  -> 24x24 * 16
        x = self.pool(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = (F.relu(self.conv2_1_bn(self.conv2_1(x))))
        x = self.pool(F.relu(self.conv2_2_bn(self.conv2_2(x))))

        x = (F.relu(self.conv3_1_bn(self.conv3_1(x))))
        x = (F.relu(self.conv3_2_bn(self.conv3_2(x))))
        x = self.pool(F.relu(self.conv3_3_bn(self.conv3_3(x))))

        x = (F.relu(self.conv4_1_bn(self.conv4_1(x))))
        x = (F.relu(self.conv4_2_bn(self.conv4_2(x))))
        x = self.pool(F.relu(self.conv4_3_bn(self.conv4_3(x))))
        # flatten image input
        # print(x.shape)
        x = x.view(-1, 512 * 3 * 3)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.dense1_bn(self.fc1(x)))
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.dense2_bn(self.fc2(x)))
        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function
        x = self.fc3(x)
        return x


