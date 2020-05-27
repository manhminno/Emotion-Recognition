import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms


#Transform
transform_train = transforms.Compose([transforms.Resize(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_val = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Encoder
label2id = {
    0: 'Gian giu',
    1: 'Vui ve',
    2: 'Binh thuong',
    3: 'Buon chan',
    4: 'Wow'
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #Feature extract
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #16


        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #8

        #FC
        self.fc1 = nn.Linear(in_features=4096, out_features=1024) #8x8x64
        self.relu = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=5)


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool2(out)
      
        #Flatten()
        out = out.view(-1, 4096) # 6x6x128 = 4608

        #FC 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc1(out)

        #Out
        out = self.fc2(out)

        return out