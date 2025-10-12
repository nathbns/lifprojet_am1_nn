import torch
import torch.nn as nn


class CNN(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bn_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn_act)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNN(channels, channels // 2, kernel_size=1),
                    CNN(channels // 2, channels, kernel_size=3, padding=1),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class ScalePrediction(nn.Module):
    """Scale prediction block for YOLO output"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNN(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNN(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, bn_act=False),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    """YOLOv3 architecture with explicit layers"""
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv1 = CNN(in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = CNN(32, 64, kernel_size=3, stride=2, padding=1)
        self.residual1 = ResidualBlock(64, num_repeats=1)
        
        self.conv3 = CNN(64, 128, kernel_size=3, stride=2, padding=1)
        self.residual2 = ResidualBlock(128, num_repeats=2)
        
        self.conv4 = CNN(128, 256, kernel_size=3, stride=2, padding=1)
        self.residual3 = ResidualBlock(256, num_repeats=8)  
        
        self.conv5 = CNN(256, 512, kernel_size=3, stride=2, padding=1)
        self.residual4 = ResidualBlock(512, num_repeats=8)  
        
        self.conv6 = CNN(512, 1024, kernel_size=3, stride=2, padding=1)
        self.residual5 = ResidualBlock(1024, num_repeats=4)

        
        self.conv7 = CNN(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv8 = CNN(512, 1024, kernel_size=3, stride=1, padding=1)
        self.residual6 = ResidualBlock(1024, num_repeats=1)
        self.conv9 = CNN(1024, 512, kernel_size=1, stride=1, padding=0)
        self.scale_pred1 = ScalePrediction(512, num_classes=num_classes)
        
        self.conv10 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv11 = CNN(768, 256, kernel_size=1, stride=1, padding=0)
        self.conv12 = CNN(256, 512, kernel_size=3, stride=1, padding=1)
        self.residual7 = ResidualBlock(512, num_repeats=1)
        self.conv13 = CNN(512, 256, kernel_size=1, stride=1, padding=0)
        self.scale_pred2 = ScalePrediction(256, num_classes=num_classes)
        
        self.conv14 = CNN(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv15 = CNN(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv16 = CNN(128, 256, kernel_size=3, stride=1, padding=1)
        self.residual8 = ResidualBlock(256, num_repeats=1)
        self.conv17 = CNN(256, 128, kernel_size=1, stride=1, padding=0)
        self.scale_pred3 = ScalePrediction(128, num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.residual1(x)
        
        x = self.conv3(x)
        x = self.residual2(x)
        
        x = self.conv4(x)
        route1 = self.residual3(x)  
        
        x = self.conv5(route1)
        route2 = self.residual4(x)  
        
        x = self.conv6(route2)
        x = self.residual5(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.residual6(x)
        x = self.conv9(x)
        out1 = self.scale_pred1(x)
        
        x = self.conv10(x)
        x = self.upsample1(x)
        x = torch.cat([x, route2], dim=1)  
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.residual7(x)
        x = self.conv13(x)
        out2 = self.scale_pred2(x)
        
        x = self.conv14(x)
        x = self.upsample2(x)
        x = torch.cat([x, route1], dim=1)  
        
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.residual8(x)
        x = self.conv17(x)
        out3 = self.scale_pred3(x)
        
        return [out1, out2, out3]