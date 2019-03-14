import torch.nn as nn
from ops_torch import gaussian_blur
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=48, kernel_size=11, stride=4, padding=5)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()

        self.leaky = nn.LeakyReLU(negative_slope=0.2)

        self._init_weight()

    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, preprocess = 'gray'):

        if preprocess == 'gray':
            print("Discriminator-texture")
            image_processed = 0.299*image[:,0,:,:] + 0.587*image[:,1,:,:] + 0.114*image[:,2,:,:]
            image_processed = image_processed.unsqueeze(1)

        elif preprocess == 'blur':
            print("Discriminator-color (blur)")
            conv_filter = torch.Tensor([[0.299, -0.14714119, 0.61497538], [0.587, -0.28886916, -0.51496512], [0.114, 0.43601035, -0.10001026]])
            conv_filter = conv_filter.unsqueeze(2)
            conv_filter = conv_filter.unsqueeze(3)
            
            if torch.cuda.is_available():
                conv_filter = conv_filter.cuda()
            
            image = F.conv2d(image, conv_filter)            
            image_processed = gaussian_blur(image)

        else:
            print("Discriminator-color (none)")
            image_processed = image

        temp = self.conv1(image_processed)
        temp = self.leaky(temp)

        temp = self.conv2(temp)
        temp = self.bn2(temp)
        temp = self.leaky(temp)
        
        temp = self.conv3(temp)
        temp = self.bn3(temp)
        temp = self.leaky(temp)
        
        temp = self.conv4(temp)
        temp = self.bn4(temp)
        temp = self.leaky(temp)
        
        temp = self.conv5(temp)
        temp = self.bn5(temp)
        temp = self.leaky(temp)

        temp = temp.view(temp.size(0), -1)
        temp = self.fc1(temp)
        temp = self.leaky(temp)
        
        logits = self.fc2(temp)
        probability = self.sigmoid(logits)

        return logits, probability
