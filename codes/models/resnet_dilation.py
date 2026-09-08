#!/usr/bin/env python
# coding: utf-8
#
# This code is based on torchvison resnet
# URL: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)  # 输出形状 [B, out_planes, H', W']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)  # 输出形状 [B, out_planes, H', W']


class BasicBlock(nn.Module):
    expansion = 1  # 输出通道膨胀倍数=1

    def __init__(self, inplanes, planes, stride, dilation, downsample=None):
        # 定义变量：inplanes, planes, stride, dilation, downsample
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3×3卷积，输出 [B, planes, H', W']
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, dilation, dilation)  # 3×3空洞卷积，输出 [B, planes, H', W']
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 旁路备份

        out = self.conv1(x)  # [B, planes, H', W']
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # [B, planes, H', W']
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 旁路下采样，形状与out一致

        out += residual  # 残差相加
        out = self.relu(out)

        return out  # 输出 [B, planes, H', W']


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道膨胀倍数=4

    def __init__(self, inplanes, planes, stride, dilation, downsample=None, expansion=4):
        # 定义变量：inplanes, planes, stride, dilation, downsample, expansion
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = conv1x1(inplanes, planes)  # 1×1降通道，输出 [B, planes, H, W]
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)  # 3×3空洞卷积，输出 [B, planes, H', W']
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 1×1升通道，输出 [B, planes*expansion, H', W']
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 旁路备份

        out = self.conv1(x)  # [B, planes, H, W]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # [B, planes, H', W']
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # [B, planes*expansion, H', W']
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 旁路下采样，形状与out一致

        out += residual  # 残差相加
        out = self.relu(out)

        return out  # 输出 [B, planes*expansion, H', W']


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, num_classes=1000, input_channels=3):
        # 定义变量：block, layers, output_stride, num_classes, input_channels
        super(ResNet, self).__init__()
        if output_stride == 8:
            stride = [1, 2, 1, 1]  # 各阶段stride
            dilation = [1, 1, 2, 2]  # 对应空洞率
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 输出 [B, 64, H/2, W/2]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出 [B, 64, H/4, W/4]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0], dilation=dilation[0])  # 输出 [B, 64*expansion, H/4, W/4]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1], dilation=dilation[1])  # 输出 [B, 128*expansion, H/8, W/8]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2], dilation=dilation[2])  # 输出 [B, 256*expansion, H/16, W/16]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3], dilation=dilation[3])  # 输出 [B, 512*expansion, H/32, W/32]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，输出 [B, 512*expansion, 1, 1]
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类层

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, dilation):
        # 定义变量：block, planes, blocks, stride, dilation
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 1×1下采样，输出 [B, planes*expansion, H', W']
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))  # 第一个block负责下采样/降维
        self.inplanes = planes * block.expansion  # 更新通道数
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilation))  # 后续block stride=1

        return nn.Sequential(*layers)  # 封装为nn.Sequential

    def forward(self, x):
        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, H/4, W/4]

        x = self.layer1(x)  # [B, 64*expansion, H/4, W/4]
        x = self.layer2(x)  # [B, 128*expansion, H/8, W/8]
        x = self.layer3(x)  # [B, 256*expansion, H/16, W/16]
        x = self.layer4(x)  # [B, 512*expansion, H/32, W/32]

        x = self.avgpool(x)  # [B, 512*expansion, 1, 1]
        x = x.view(x.size(0), -1)  # 展平 [B, 512*expansion]
        x = self.fc(x)  # [B, num_classes]

        return x


# 以下函数均返回对应ResNet模型，并可选择加载ImageNet预训练权重
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  # 各阶段block数
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model