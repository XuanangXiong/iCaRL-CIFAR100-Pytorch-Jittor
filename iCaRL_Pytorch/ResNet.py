import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam']

# 3x3卷积层，保持维度不变
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
'''
nn.Conv2d(
    in_planes,        # 输入特征图通道数（C_in）
    out_planes,       # 输出特征图通道数（C_out），等价于卷积核个数
    kernel_size=3,    # 卷积核空间尺寸
    stride=stride,    # 步长
    padding=1,        
    bias=False,       # False一般在后面加上BatchNorm
    dilation=1,       # 空洞卷积孔洞大小，默认 1 即普通卷积
    groups=1,         # 分组卷积，默认 1 不分组
)
weight.shape = (out_planes, in_planes, kernel_h, kernel_w)
bias.shape   = (out_planes,)            # if bias=True
'''

# CBAM（Convolutional Block Attention Module）机制，包括通道注意力和空间注意力

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局池化操作（压成1x1的数字，注意在通道间的差别上）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 激活函数与两个卷积（MLP）
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    # 通道注意力权重
    def forward(self, x):
        # input (batch_size,Channels_num,hight,width)
        # (32,256,H,W)->pool(32,256,1,1)->fc1(32,16,1,1)->relu->fc2(32,256,1,1)->sigmoid
        # fc的核大小为1本质是全连接，但是比全连接方便
        # 先降后升维度节省参数，中间能加上RelU非线性增强，fc1压缩迫使网络学习最重要的信息
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        '''
        kernel=7 → padding=3  (7-1)/2 = 3
        kernel=3 → padding=1  (3-1)/2 = 1
        '''

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input (batch_size,Channels_num,hight,width)
        # (32,256,H,W)->avg & max(32,1,H,W)->cat(32,2,H,W)->conv1(32,1,H,W)->sigmoid

        #print('输入的shape为:'+str(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True) # 通道维度平均
        #print('avg_out的shape为:' + str(avg_out.shape))
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 通道维度最大值
        #print('max_out的shape为:' + str(max_out.shape))
        x = torch.cat([avg_out, max_out], dim=1) # 拼接为2通道
        x = self.conv1(x) # 卷积到1通道
        return self.sigmoid(x) # 空间注意力权重

# ResNet 18/34 使用
class BasicBlock(nn.Module):
    expansion = 1 # 输出通道数 = 输入通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        #y=torch.norm(x,p=2,dim=1,keepdim=True)
        #x/=y
        #x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

