import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math

# 预训练权重下载地址
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth'
}

def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    # 定义变量：pretrained
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)  # 构建Res2Net-50
    if pretrained:
        # 本地权重路径加载（未使用url）
        ckpt_path = r'D:\\study\\qjxzxjc\\TDFNet\\CSMA-Net_models\\pretrain\\RCRNet_res2_pretrain.pth'
        model_state = torch.load(ckpt_path, map_location='cpu')  # 加载cpu字典
        model.load_state_dict(model_state, strict=False)  # 载入模型

    return model  # 返回模型实例


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        # 定义变量：block, layers, baseWidth, scale, num_classes
        self.inplanes = 64  # 初始通道数
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth  # 基础宽度
        self.scale = scale  # 分支数量
        # 三段3×3卷积替代ResNet的7×7，输出形状 [B, 64, H/2, W/2]（步长2+1+1）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)  # [B, 64, H/2, W/2]
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出 [B, 64, H/4, W/4]
        self.layer1 = self._make_layer(block, 64, layers[0])  # 构建layer1，输出 [B, 256, H/4, W/4]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 输出 [B, 512, H/8, W/8]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 输出 [B, 1024, H/16, W/16]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 输出 [B, 2048, H/32, W/32]
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 未使用（注释掉）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 未使用

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 定义变量：block, planes, blocks, stride
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 下采样分支：平均池化+1×1卷积，输出 [B, planes*expansion, H', W']
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))  # 第一个block负责下采样/降维
        self.inplanes = planes * block.expansion  # 更新通道数
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))  # 后续block stride=1

        return nn.Sequential(*layers)  # 封装为nn.Sequential

    def forward(self, x):
        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)    # torch.Size([1, 64, 176, 176])  （示例）
        x = self.maxpool(x)  # [B, 64, H/4, W/4]

        l1 = self.layer1(x)  # [B, 256, H/4, W/4]
        l2 = self.layer2(l1)  # [B, 512, H/8, W/8]
        l3 = self.layer3(l2)  # [B, 1024, H/16, W/16]
        l4 = self.layer4(l3)  # [B, 2048, H/32, W/32]

        # x = self.avgpool(x)  # 未使用
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)       # 未使用

        return l4, l3, l2, l1  # 返回4级特征，解码用


class Bottle2neck(nn.Module):
    expansion = 4  # 输出通道膨胀倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        # 定义变量：inplanes, planes, stride, downsample, baseWidth, scale, stype
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality (未乘expansion)
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))  # 每个分支基础通道数
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)  # 1×1降/升通道，输出 [B, width*scale, H, W]
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1  # 仅1个分支
        else:
            self.nums = scale - 1  # 剩余分支数
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)  # 下采样用
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))  # 3×3空洞卷积
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)  # 包装为ModuleList
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)  # 1×1升通道，输出 [B, planes*expansion, H', W']
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 旁路下采样
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x  # 旁路备份

        out = self.conv1(x)  # [B, width*scale, H, W]
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)  # 按通道均分scale份，list每个形状 [B, width, H, W]
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]  # 第一个分支直接取
            else:
                sp = sp + spx[i]  # 后续分支累加前一分支输出
            sp = self.convs[i](sp)  # 3×3卷积，stride同block stride
            sp = self.relu(self.bns[i](sp))  # 激活
            if i == 0:
                out = sp  # 第一个分支直接赋值
            else:
                out = torch.cat((out, sp), 1)  # 沿通道拼接
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)  # 加上未处理最后一个分支
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)  # 最后一个分支先下采样再拼接

        out = self.conv3(out)  # [B, planes*expansion, H', W']
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 旁路下采样，形状与out一致

        out += residual  # 残差相加
        out = self.relu(out)

        return out  # 最终输出 [B, planes*expansion, H', W']