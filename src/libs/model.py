# pip imports
import torch.nn as nn
import torch

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        # nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x

class ChessModelConv2d(nn.Module):
    def __init__(self, num_classes):
        super(ChessModelConv2d, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv_1 = nn.Conv2d(14, 128, kernel_size=3, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2d_1 = nn.Dropout2d(0.05)
        # NOTE: no max pooling layers ??
        # NOTE: no dropout layers ??
        # self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.dropout2d_2 = nn.Dropout2d(0.05)

        self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(8*8 * 64, num_classes)
        # self.dropout_1 = nn.Dropout(0.01)
        # self.fc2 = nn.Linear(256, num_classes)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        # nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        # x = self.maxpool(x)
        # x = self.dropout2d_1(x)
        # x = self.relu(self.conv_2(x))
        # x = self.maxpool(x)
        # x = self.dropout2d_2(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.dropout_1(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x

class ChessModelLinear(torch.nn.Module):

    def __init__(self, num_classes):
        super(ChessModelLinear, self).__init__()
        self.INPUT_SIZE = 14*8*8 
        # self.INPUT_SIZE = 7*7*13 #NOTE changing input size for using cnns
        self.OUTPUT_SIZE = num_classes # = number of unique moves (action space)

        #can try to add CNN and pooling here (calculations taking into account spacial features)

        #input shape for sample is (8,8,14), flattened to 1d array of size 896
        # self.cnn1 = nn.Conv3d(4,4,(2,2,4), padding=(0,0,1))
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.linear4 = torch.nn.Linear(1000, 200)
        self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)
        self.softmax = torch.nn.Softmax(1) #use softmax as prob for each move, dim 1 as dim 0 is the batch dimension

    def forward(self, x): #x.shape = (batch size, 896)
        x = x.to(torch.float32)
        # x = self.cnn1(x) #for using cnns
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        # x = self.softmax(x) #do not use softmax since you are using cross entropy loss
        return x