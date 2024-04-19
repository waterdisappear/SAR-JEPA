import torch
from torch import nn
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class A_ConvNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super(A_ConvNet, self).__init__()
        # ConvTranspose2d output = (input-1)stride+outputpadding -2padding+kernelsize
        self.conv1 = nn.Conv2d(in_ch, 16, 5, 1, 0)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 6, 1, 0)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 0)
        self.drop = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(128, num_classes, 3, 1, 0)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.Relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.Relu(x)
        # x = self.drop(x)
        out = self.conv5(x)
        return out.squeeze()

class A_ConvNet_BN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super(A_ConvNet_BN, self).__init__()
        # ConvTranspose2d output = (input-1)stride+outputpadding -2padding+kernelsize
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 16, 5, 1, 0), nn.BatchNorm2d(16))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 0), nn.BatchNorm2d(32))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 6, 1, 0), nn.BatchNorm2d(64))
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 0), nn.BatchNorm2d(128))
        # self.drop = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(128, num_classes, 3, 1, 0)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.Relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.Relu(x)
        # x = self.drop(x)
        out = self.conv5(x)
        return out.squeeze(-1).squeeze(-1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_planes, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc1(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(2)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def atten(x, con, ca ,sa):
    x = con(x)
    x = x*ca(x)
    y = x*sa(x)
    return y


class AMCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(AMCNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128
        self.num = num_classes
        # 3 * 128 * 128
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (3, 3)), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (7, 7)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, (5, 5)), nn.BatchNorm2d(128), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, (5, 5)), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, (6, 6)), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv8 = nn.Conv2d(64, self.num, (3, 3))

        self.CA1 = ChannelAttention(16)
        self.SA1 = SpatialAttention()
        self.CA2 = ChannelAttention(32)
        self.SA2 = SpatialAttention()
        self.CA3 = ChannelAttention(64)
        self.SA3 = SpatialAttention()
        self.CA4 = ChannelAttention(128)
        self.SA4 = SpatialAttention()

        self.CA5 = ChannelAttention(256)
        self.SA5 = SpatialAttention()
        self.CA6 = ChannelAttention(128)
        self.SA6 = SpatialAttention()
        self.CA7 = ChannelAttention(64)
        self.SA7 = SpatialAttention()

        self.maxpool = nn.MaxPool2d((2,2))


    def forward(self, x):
        x1 = atten(x, self.conv1, self.CA1, self.SA1)
        x2 = atten(x1, self.conv2, self.CA2, self.SA2)
        x3 = atten(x2, self.conv3, self.CA3, self.SA3)
        x4 = atten(x3, self.conv4, self.CA4, self.SA4)
        x4 = self.maxpool(x4)

        x5 = atten(x4, self.conv5, self.CA5, self.SA5)
        x5 = self.maxpool(x5)
        x6 = atten(x5, self.conv6, self.CA6, self.SA6)
        x6 = self.maxpool(x6)
        x7 = atten(x6, self.conv7, self.CA7, self.SA7)

        out = self.conv8(x7).squeeze(-1).squeeze(-1)
        return out


class BNACNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(BNACNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        # batsize,channel,height,length 20*3*128*128
        self.num = num_classes
        # 3 * 128 * 128
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (3, 3)), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (7, 7)), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, (5, 5)), nn.BatchNorm2d(128), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, (5, 5)), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, (6, 6)), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, (5, 5)), nn.BatchNorm2d(64), nn.ReLU())
        self.conv8 = nn.Conv2d(64, self.num, (3, 3))

        self.maxpool = nn.MaxPool2d((2,2))


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = self.maxpool(x4)

        x5 = self.conv5(x4)
        x5 = self.maxpool(x5)
        x6 = self.conv6(x5)
        x6 = self.maxpool(x6)
        x7 = self.conv7(x6)

        out = self.conv8(x7).squeeze(-1).squeeze(-1)
        return out


class MVGGNet(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(MVGGNet, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.MaxPool2d(kernel_size=2, stride=1)
        model = models.vgg11(pretrained=False)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            # nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class ResNet_18(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_18, self).__init__()

        model = models.resnet18(pretrained=False)
        # print(model)
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class ResNet_34(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_34, self).__init__()

        model = models.resnet34(pretrained=False)
        # print(model)
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class ResNet_50(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_50, self).__init__()

        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        self.model = model

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class convnext_1(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(convnext_1, self).__init__()

        model = models.convnext_tiny(pretrained=True)
        # print(model)
        model.classifier[2] = nn.Linear(768, num_classes)
        self.model = model

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class efficientnet_b0(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(efficientnet_b0, self).__init__()

        model = models.efficientnet_b0()
        print(model)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model = model


    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


class efficientnet_b1(torch.nn.Module):
    def __init__(self, feature_extract=True, num_classes=3):
        super(efficientnet_b1, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        model = models.efficientnet_b1()
        print(model)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model = model


    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        out = self.model(x)
        return out


# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('..')
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



def crop(x1, x2):
    '''
        conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
    '''

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                diffY // 2, diffY - diffY//2))

    x = torch.cat([x2, x1], dim=1)
    return x


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.padding = nn.ReflectionPad2d(padding)
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.padding(x)
        return F.relu(self.seq(x))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, padding=2, bias=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            BN_Conv2d(in_ch, out_ch, 5, stride=1, padding=padding, bias=bias),
            BN_Conv2d(out_ch, out_ch, 5, stride=1, padding=padding, bias=bias)
        )

    def forward(self, input):
        return self.conv(input)


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        # self.conv = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=out_channels * num_conv_units,
        #                       kernel_size=kernel_size,
        #                       stride=stride)
        self.conv = BN_Conv2d(in_channels, out_channels * num_conv_units, kernel_size, stride=stride, padding=0, bias=False)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim),
                              requires_grad=True)
    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v


def MLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_classes=3):
        super(Unet, self).__init__()
        base_channel = 32
        avgpool = 8

        self.conv1 = DoubleConv(in_ch, base_channel)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_channel, base_channel * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_channel * 2, base_channel * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_channel * 4, base_channel * 8)

        self.avg = nn.AdaptiveMaxPool2d((avgpool, avgpool))
        # Primary capsule
        self.linear = nn.Sequential(nn.Linear(base_channel * (1+2+4+8)*8*8, 512, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, num_classes, bias=False))

        self.primary_caps = PrimaryCaps(num_conv_units=16,
                                        in_channels=base_channel * 15,
                                        out_channels=8,
                                        kernel_size=3,
                                        stride=1)
        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=16 * 6 * 6,
                                    num_caps=num_classes,
                                    dim_caps=8,
                                    num_routing=4)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        feature1 = torch.cat([self.avg(c1), self.avg(c2), self.avg(c3), self.avg(c4)], 1)
        # feature2 = self.primary_caps(feature1)
        # out = self.digit_caps(feature2)
        # logits = torch.norm(out, dim=-1)

        logits = self.linear(feature1.view(x.size(0), -1))
        return logits

