import torch 
import torch.nn as nn

class CNN(nn.Module):
    """
    **kwargs tous les autre args, sous forme de dict,
    couche de convolution, bias=False parce que l'on batchNorm (il a son propre biais),
    leaky relue: si x > 0 -> x, sinon -> 0.1 * x
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelue = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelue(self.batchnorm(self.conv(x)))


class Yolo_V1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(Yolo_V1, self).__init__()

        # Darknet model, mais from scratch
        self.conv1 = CNN(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = CNN(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = CNN(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = CNN(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = CNN(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = CNN(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc répété 4 fois: (1x1 256) -> (3x3 512)
        self.conv7 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv8 = CNN(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv10 = CNN(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv12 = CNN(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv14 = CNN(256, 512, kernel_size=3, stride=1, padding=1)

        self.conv15 = CNN(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv16 = CNN(512, 1024, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloc répété 2 fois: (1x1 512) -> (3x3 1024)
        self.conv17 = CNN(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv18 = CNN(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv19 = CNN(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv20 = CNN(512, 1024, kernel_size=3, stride=1, padding=1)

        self.conv21 = CNN(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv22 = CNN(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.conv23 = CNN(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv24 = CNN(1024, 1024, kernel_size=3, stride=1, padding=1)

        # Head du modele
        S, B, C = split_size, num_boxes, num_classes
        self.fc1 = nn.Linear(1024 * S * S, 496)
        self.dropout = nn.Dropout(0.0)
        self.leaky = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(496, S * S * (C + B * 5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool4(x)

        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.leaky(x)
        x = self.fc2(x)
        return x