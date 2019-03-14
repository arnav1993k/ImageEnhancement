import torch.nn as nn
from block_torch import InceptionBlock
import torch

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self):
        super(Generator, self).__init__()
        
        self.depth1 = nn.Conv2d(in_channels = 3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.point1 = nn.Conv2d(in_channels = 3, out_channels=50, kernel_size=1)

        self.relu = nn.ReLU()
        self.inception = InceptionBlock(in_channels=50)

        self.depth2 = nn.Conv2d(in_channels=146, out_channels=146, kernel_size=3, stride=1, padding=1, groups=146)
        self.point2 = nn.Conv2d(in_channels=146, out_channels=64, kernel_size=1)

        self.depth3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.point3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.depth4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.point4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)

        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, image, scope = 'generator'):

        temp = self.depth1(image)
        temp = self.point1(temp)
        temp = self.relu(temp)

        if scope == 'generator':
            temp = self.inception(temp)
        
        temp = self.depth2(temp)
        temp = self.point2(temp)
        temp = self.relu(temp)

        if scope == 'generator':
            temp = self.depth3(temp)
            temp = self.point3(temp)
            temp = self.bn3(temp)
            temp = self.relu(temp)

        temp = self.depth4(temp)
        temp = self.point4(temp)

        if scope != 'generator':
            temp = self.bn4(temp)

        temp = self.relu(temp) 
        temp = self.conv(temp)

        return temp
        


        # temp = self.depth1(image)
        # temp = self.point1(temp)
        # temp = self.relu(temp)

        # if scope == 'generator':
        #     temp = self.inception(temp)

        #     temp = self.depth2(temp)
        #     temp = self.point2(temp)
        #     temp = self.relu(temp)

        #     temp = self.depth3(temp)
        #     temp = self.point3(temp)
        #     temp = self.bn3(temp)
        #     temp = self.relu(temp)

        #     temp = self.depth4(temp)
        #     temp = self.point4(temp)
        # else:
        #     temp = self.depth2(temp)
        #     temp = self.point2(temp)
        #     temp = self.relu(temp)

        #     temp = self.depth4(temp)
        #     temp = self.point4(temp)
        #     temp = self.bn4(temp)

        # temp = self.relu(temp)
        # temp = self.conv(temp)

        # return temp