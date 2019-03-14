import torch.nn as nn
import torch

class InceptionBlock(nn.Module):
    """docstring for Inception_Block"""
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        self.in_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels = self.in_channels, out_channels=32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels = self.in_channels, out_channels=32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride=1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=5, stride=1, padding = 2)

        self.relu = nn.ReLU()
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feature_in):
        
        simple_1 = self.conv1(feature_in)
        simple_2 = self.conv2(feature_in)
        
        simple_3 = self.conv3(feature_in)
        simple_3 = self.relu(simple_3)

        filter_1 = self.conv4(simple_1)
        filter_1 = self.relu(filter_1)

        filter_2 = self.conv5(simple_2)
        filter_2 = self.relu(filter_2)

        stack = torch.cat((simple_3, filter_1, filter_2, feature_in), dim=1)

        return stack
        

class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feature_in):
        
        temp = self.conv1(feature_in)
        temp = self.relu(temp)
        temp = self.conv2(temp)
        temp = self.relu(temp)

        temp = temp + feature_in
        
        return temp
        