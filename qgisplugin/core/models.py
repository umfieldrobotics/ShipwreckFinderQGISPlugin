from ..safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()

import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet101, resnet50, resnet18
import torchvision.models as models
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functools import partial

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet


from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.res_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.res_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SEBasicBlock(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBasicBlock(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, prev_lay_ch, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(prev_lay_ch, out_ch, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class OutputConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputConv, self).__init__()
        self.conv = SEBasicBlock(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels 

        self.l1 = nn.Linear(in_channels, in_channels//8)
        self.l2 = nn.Linear(in_channels//8, in_channels//16)
        self.out = nn.Linear(in_channels//16, 1)


    
    def forward(self, x):
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = self.out(x)
        return x



class ProxyModel(nn.Module):

    def __init__(self, in_channels, classes, regression=False, jigsaw=False, permutations=5):
        super(ProxyModel, self).__init__()
        self.resnet_encoder = models.resnet34(pretrained=True)
        self.activations = {}
        self.regression = regression
        self.resnet_encoder.conv1.register_forward_hook(self.get_activation('conv1'))
        self.resnet_encoder.layer1.register_forward_hook(self.get_activation('layer1'))
        self.resnet_encoder.layer2.register_forward_hook(self.get_activation('layer2'))
        self.resnet_encoder.layer3.register_forward_hook(self.get_activation('layer3'))
        self.resnet_encoder.layer4.register_forward_hook(self.get_activation('layer4'))
        self.jigsaw = jigsaw

        self.flatten = torch.nn.Flatten()
        self.l1_avg = nn.AvgPool1d(8)
        self.l2_avg = nn.AvgPool1d(4)
        self.l3_avg = nn.AvgPool1d(2)
        self.linear_1 = nn.Linear(98304, 2)
        if self.regression:   
          self.linear_2 = nn.Linear(98304, 6)
        if self.jigsaw: 
            # self.linear_2 = nn.Sequential(nn.Linear(41472, 4096), 
            #                                 nn.Linear(4096, 100),
            #                                 nn.Linear(100, permutations)) 
            self.linear_2 = nn.Linear(41472, permutations)
        self.u1 = DoubleConv(512, 1024)
        self.u2 = Up(512+256, 512, 1024)
        self.u3 = Up(256+128, 256, 512)
        self.u4 = Up(64+128, 128, 256)
        self.u5 = Up(64+64, 64, 128)
        self.u6 = Up(32+3, 32, 64)
        self.outc = OutputConv(32, classes)
        


    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            try:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
            except:
                print("failed")
                pass

    def get_patch_features(self, x):
        pred = self.resnet_encoder(x)
        l2 = self.activations["layer2"]
        l3 = self.activations["layer3"]
        return l2, l3


    def forward(self, x, no_avg=False):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        x = self.u1(l4)
        x = self.u2(x, l3)
        x = self.u3(x, l2)
        x = self.u4(x, l1)
        x = self.u5(x, c1)
        x = self.u6(x, input)
        x = self.outc(x)
        
        l4 = self.flatten(l4)
       
        if no_avg:
            latent_vect = l4
        else:
            l1 = self.l1_avg(self.flatten(l1))
            l2 = self.l2_avg(self.flatten(l2))
            l3 = self.l3_avg(self.flatten(l3))
            latent_vect = l1+l2+l3+l4

        return x, latent_vect
    
    def temporal_inference(self, x): #x is (B, 3, C, N, N)
        im1 = x[:,0]
        im2 = x[:,1]
        im3 = x[:,2]

        o1, l1 = self.forward(im1)
        o2, l2 = self.forward(im2)
        o3, l3 = self.forward(im3)
        latent_concat = torch.cat((l1, l2, l3), 1)
        seg_concat = torch.cat((o1.unsqueeze(-1), o2.unsqueeze(-1), o3.unsqueeze(-1)), dim=4)
        seg_concat = torch.permute(seg_concat, (0, 1, 4, 2, 3))
        output = self.linear_1(latent_concat)

        return output, seg_concat
    
    def temporal_regression(self, x): #x is (B, 3, C, N, N)
        if x.dim() != 5:
            x = x.unsqueeze(0)
        im1 = x[:,0]
        im2 = x[:,1]
        im3 = x[:,2]

        o1, l1 = self.forward(im1)
        o2, l2 = self.forward(im2)
        o3, l3 = self.forward(im3)
        latent_concat = torch.cat((l1, l2, l3), 1)
        seg_concat = torch.cat((o1.unsqueeze(-1), o2.unsqueeze(-1), o3.unsqueeze(-1)), dim=4)
        seg_concat = torch.permute(seg_concat, (0, 1, 4, 2, 3))
        output = self.linear_2(latent_concat)

        return output, seg_concat
    

    def jigsaw_task(self, x):
        features = []
        for i in range(9):
            o1, l1 = self.forward(x[:,i], no_avg=True)
            features.append(l1)
        features = torch.cat(features, dim=1)
        prediction = self.linear_2(features)
        return prediction
        

    def get_features(self, input): #input is (B, C, H, W)
        o1, l1 = self.forward(input)
        return l1

class Unet(nn.Module):

    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.resnet_encoder = models.resnet34()
        self.resnet_encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.activations = {}

        self.resnet_encoder.conv1.register_forward_hook(self.get_activation('conv1'))
        self.resnet_encoder.layer1.register_forward_hook(self.get_activation('layer1'))
        self.resnet_encoder.layer2.register_forward_hook(self.get_activation('layer2'))
        self.resnet_encoder.layer3.register_forward_hook(self.get_activation('layer3'))
        self.resnet_encoder.layer4.register_forward_hook(self.get_activation('layer4'))


        self.u1 = DoubleConv(512, 1024)
        self.u2 = Up(512+256, 512, 1024)
        self.u3 = Up(256+128, 256, 512)
        self.u4 = Up(64+128, 128, 256)
        self.u5 = Up(64+64, 64, 128)
        self.u6 = Up(32+in_channels, 32, 64)
        self.outc = OutputConv(32, classes)


    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            try:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
            except:
                print("failed")
                pass


    def forward(self, x):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        x = self.u1(l4)
        x = self.u2(x, l3)
        x = self.u3(x, l2)
        x = self.u4(x, l1)
        x = self.u5(x, c1)
        x = self.u6(x, input)
        x = self.outc(x)
        return x
    
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class mymodel(nn.Module):
    
    def __init__(self, classes, w1=0.3, w2=0.3, w3=0.3):
        super(mymodel, self).__init__()
        self.deformation_encoder = models.resnet34(pretrained=True)
        self.student = models.resnet34(pretrained=True)
        self.teacher = models.resnet34(pretrained=True)
        freeze_weights(self.teacher, 4)
        self.teacher.eval()

        self.activations = {}
        self.deformation_encoder.conv1.register_forward_hook(self.get_activation('def_conv1'))
        self.deformation_encoder.layer1.register_forward_hook(self.get_activation('def_layer1'))
        self.deformation_encoder.layer2.register_forward_hook(self.get_activation('def_layer2'))
        self.deformation_encoder.layer3.register_forward_hook(self.get_activation('def_layer3'))
        self.deformation_encoder.layer4.register_forward_hook(self.get_activation('def_layer4'))

        self.student.conv1.register_forward_hook(self.get_activation('student_conv1'))
        self.student.layer1.register_forward_hook(self.get_activation('student_layer1'))
        self.student.layer2.register_forward_hook(self.get_activation('student_layer2'))
        self.student.layer3.register_forward_hook(self.get_activation('student_layer3'))
        self.student.layer4.register_forward_hook(self.get_activation('student_layer4'))

        self.teacher.conv1.register_forward_hook(self.get_activation('teacher_conv1'))
        self.teacher.layer1.register_forward_hook(self.get_activation('teacher_layer1'))
        self.teacher.layer2.register_forward_hook(self.get_activation('teacher_layer2'))
        self.teacher.layer3.register_forward_hook(self.get_activation('teacher_layer3'))
        self.teacher.layer4.register_forward_hook(self.get_activation('teacher_layer4'))
       

        self.final_dec1 = Up(240, 120, 240)
        # self.u1 = DoubleConv(512, 1024)
        self.def_u2 = Up(512+1, 256, 512+1)
        self.def_u3 = Up(256+1, 128, 256)
        self.def_u4 = Up(128+1, 64, 128)
        self.def_u5 = Up(64+32, 32, 64)
        self.def_u6 = Up(16+3, 16, 32)
        self.def_outc = OutputConv(16, 10+20)

        self.student_u2 = Up(512, 256, 512)
        self.student_u3 = Up(256, 128, 256)
        self.student_u4 = Up(128, 64, 128)
        self.student_u5 = Up(64+32, 32, 64)
        self.student_u6 = Up(16+3, 16, 32)
        self.student_outc = OutputConv(16, 1)
        self.student_conv1 = nn.Conv2d(480, 960, 1, 1)

        self.attn1 = Self_Attn(128, nn.Softmax())
        self.attn2 = Self_Attn(256, nn.Softmax())
        self.attn3 = Self_Attn(512, nn.Softmax())
        self.attn4 = Self_Attn(1024, nn.Softmax())
        self.attn_layers = [self.attn1, self.attn2, self.attn3, self.attn4]

        self.small_dec = nn.Sequential(nn.Conv2d(960+4, 480, 3, 1, padding=1), 
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                                        nn.Conv2d(480, 128, 3, 1, padding=1), 
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                                        nn.Conv2d(128, 1, 1, 1))
        self.out_conv = nn.Conv2d(34, 1, 1)
        # self.out_conv = SEBasicBlock(34,1)
        self.seg2 = Up(256+1024, 256, 512)
        self.seg3 = Up(128+512, 128, 256)
        self.seg4 = Up(64+256, 64, 128)
        self.seg5 = Up(32+128, 32, 64)
        self.seg6 = Up(16+3, 16, 32)
        self.seg_outc = OutputConv(16, 1)
        self.abn1 = nn.BatchNorm2d(1)
        self.abn2 = nn.BatchNorm2d(1)
        self.abn3 = nn.BatchNorm2d(1)
        self.abn4 = nn.BatchNorm2d(1)
        self.abns = [self.abn1, self.abn2, self.abn3, self.abn4]

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook
    
    def forward(self, x):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        x = self.u1(l4)
        x = self.u2(x, l3)
        x = self.u3(x, l2)
        x = self.u4(x, l1)
        x = self.u5(x, c1)
        x = self.u6(x, input)
        seg_out = self.outc(x)
        
        embedding = self.embedder(resnet_out)
        reverse_embed = self.grl(embedding)
        domain_out = self.domain_classifier(reverse_embed)

        return seg_out, domain_out, embedding

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            try:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    
                    param = param.data
                own_state[name].copy_(param)
                print("loaded success:", name)
            except:
                print("failed")
                pass

    def deformation_branch(self, broken_img):
        """
        Runs inference of the deformation branch
        Inputs: augmented image (B,C,H,W), broken image (B,C,H,W)
        Outputs: Unwarped prediction of shipwreck (B,C,H,W), features from deformation encoder(B,C',H',W')
        """
        resnet_out = self.deformation_encoder(broken_img)
        c1 = self.activations['def_conv1']
        l1 = self.activations['def_layer1']
        l2 = self.activations['def_layer2']
        l3 = self.activations['def_layer3']
        l4 = self.activations['def_layer4']
        l1up = self.def_u2(l4, l3)
        l2up = self.def_u3(l1up, l2)
        l3up = self.def_u4(l2up, l1)
        l4up = self.def_u5(l3up, c1)
        l5up = self.def_u6(l4up, broken_img)
        def_pred = self.def_outc(l5up)
        mag_pred = def_pred[:,0:10]
        angle_pred = def_pred[:,10:]

        res = torch.nn.Upsample(size=tuple(l1.shape[-2:]), mode='bilinear')
        l2_resize = res(l2)
        l3_resize = res(l3)
        l4_resize = res(l4)

        deformation_feats = torch.cat([l1, l2_resize, l3_resize, l4_resize], dim=1)

        return mag_pred, angle_pred, deformation_feats

    def student_branch(self, aug_img):
        resnet_out = self.student(aug_img)
        c1 = self.activations['student_conv1']
        l1 = self.activations['student_layer1']
        l2 = self.activations['student_layer2']
        l3 = self.activations['student_layer3']
        l4 = self.activations['student_layer4']
        l1up = self.student_u2(l4, l3)
        l2up = self.student_u3(l1up, l2)
        l3up = self.student_u4(l2up, l1)
        l4up = self.student_u5(l3up, c1)
        l5up = self.student_u6(l4up, aug_img)
        student_pred = self.student_outc(l5up)

        res = torch.nn.Upsample(size=tuple(l1.shape[-2:]), mode='bilinear')
        l1up_resize = res(l1up)
        l2up_resize = res(l2up)
        l3up_resize = res(l3up)
        l4up_resize = res(l4up)
        student_feats = torch.cat([l4up_resize, l3up_resize, l2up_resize, l1up_resize], dim=1) #(N,C, H, W)
        student_feats = self.student_conv1(student_feats)
        student_terrain_prototype = student_feats.mean(dim=(2,3), keepdim=True) #(N,C,1,1)

        return (l3up, l2up, l1up, l4)

    def teacher_branch(self, terrain, aug_img):
        resnet_out = self.teacher(terrain)
        gt_c1 = self.activations['teacher_conv1']
        gt_l1 = self.activations['teacher_layer1']
        gt_l2 = self.activations['teacher_layer2']
        gt_l3 = self.activations['teacher_layer3']
        gt_l4 = self.activations['teacher_layer4']


        #produce aug_img feature map
        resnet_out = self.teacher(aug_img)
        aug_c1 = self.activations['teacher_conv1']
        aug_l1 = self.activations['teacher_layer1']
        aug_l2 = self.activations['teacher_layer2']
        aug_l3 = self.activations['teacher_layer3']
        aug_l4 = self.activations['teacher_layer4']


        return {"gt_feats": (gt_l1, gt_l2, gt_l3, gt_l4), "aug_feats": (aug_l1, aug_l2, aug_l3, aug_l4)}
        

    def predict_segmentation(self, terrain, broken_img):
        N, C, H, W = broken_img.shape
        # mag_pred, angle_pred, def_feats = self.deformation_branch(broken_img)
        stud_feats = self.student_branch(broken_img)
        teacher_dict = self.teacher_branch(terrain, broken_img)

        #(N,C,H,W), want norm along C axis
        # attns = []
        anomalies = []
        student_prototypes = []
        teacher_prototypes = []
        anom_res = torch.nn.Upsample(size=(H, W), mode='bilinear')
        big_anomalies = []
        for i in range(4):
            C = stud_feats[i].shape[1]

            student_terrain_prototype = stud_feats[i].mean(dim=(2,3), keepdim=True)
            student_prototypes.append(student_terrain_prototype)
            teacher_terrain_prototype = teacher_dict["gt_feats"][i].mean(dim=(2,3), keepdim=True)
            teacher_prototypes.append(teacher_terrain_prototype)

            teacher_aug_feats = teacher_dict["aug_feats"][i]
            a = student_terrain_prototype/torch.norm(student_terrain_prototype, dim=1, keepdim=True)
            b = teacher_aug_feats/torch.norm(teacher_aug_feats, dim=1, keepdim=True)
            anomaly_map = 1 - (a*b).sum(dim=1, keepdim=True)
            # anomaly_map = anomaly_map.repeat(1,C,1,1).detach() #detach anomaly map from gradients
            anomaly_map = self.abns[i](anomaly_map.detach())
            # anomaly_map = torch.zeros_like(anomaly_map.detach())
            anomalies.append(anomaly_map)
            big_anomalies.append(anom_res(anomaly_map))
        
        # seg_out = self.seg2(teacher_dict["aug_feats"][3], attns[3])
        # seg_out = self.seg3(seg_out, attns[2])
        # seg_out = self.seg4(seg_out, attns[1])
        # seg_out = self.seg5(seg_out, attns[0])
        # res = torch.nn.Upsample(size=tuple(aug_img.shape[-2:]), mode='bilinear')
        # seg_out = self.seg6(res(seg_out), aug_img)
        # seg_out = self.seg_outc(seg_out)


        resnet_out = self.deformation_encoder(broken_img)
        c1 = self.activations['def_conv1']
        l1 = self.activations['def_layer1']
        l2 = self.activations['def_layer2']
        l3 = self.activations['def_layer3']
        l4 = self.activations['def_layer4']
        l1up = self.def_u2(torch.cat([l4, anomalies[-1]], dim=1), torch.cat([l3, anomalies[-2]], dim=1))
        l2up = self.def_u3(l1up, torch.cat([l2, anomalies[-3]], dim=1))
        l3up = self.def_u4(l2up, torch.cat([l1, anomalies[-4]], dim=1))
        l4up = self.def_u5(l3up, c1)
        l5up = self.def_u6(l4up, broken_img)
        def_pred = self.def_outc(l5up)
        mag_pred = def_pred[:,0:10]
        angle_pred = def_pred[:,10:]
        anomaly_stack = torch.cat(big_anomalies, dim=1)
        payload = torch.cat([def_pred, anomaly_stack], dim=1)
        seg_pred = self.out_conv(payload)


        # attn_map = nn.functional.softmax(torch.matmul(deformation_feats.permute(0,1,3,2), teacher_aug_feats), dim=2)
        # self_attn_map = self.self_attn_conv(torch.matmul(anomaly_map, attn_map))
        #1. deformation feats 
        #2. teacher aug_feats
        #3. anomaly map
        # resnet_out = self.resnet_encoder(x)
        # input = x
        # c1 = self.activations['conv1']
        # l1 = self.activations['layer1']
        # l2 = self.activations['layer2']
        # l3 = self.activations['layer3']
        # l4 = self.activations['layer4']
        # l1up = self.seg2(l4, l3)
        # l2up = self.seg3(l1up, l2)
        # l3up = self.seg4(l2up, l1)
        # l4up = self.seg5(l3up, c1)
        # l5up = self.seg6(l4up, input)
        # seg_pred = self.seg_outc(l5up)
        
        # up = torch.nn.Upsample(size=tuple(x.shape[-2:]), mode='bilinear')
        # l1_sim = (l3up*tl1/torch.norm(l3up, dim=1, keepdim=True)/torch.norm(tl1,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l2_sim = (l2up*tl2/torch.norm(l2up, dim=1, keepdim=True)/torch.norm(tl2,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l3_sim = (l1up*tl3/torch.norm(l1up, dim=1, keepdim=True)/torch.norm(tl3,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # all_feats = torch.cat([up(l1_sim), up(l2_sim), up(l3_sim)], dim=1).detach()
        # all_feats[torch.isnan(all_feats)] = 1e-9
        # rn2_out = self.decoder(all_feats)
        # seg_pred = self.seg_outc(rn2_out)
        
        # seg_pred = self.aspp(rn2_out)
        # if torch.any(torch.isnan(seg_pred)):
        #     print("oink")


        return seg_pred, mag_pred, angle_pred, student_prototypes, teacher_prototypes
    
    def test_segmentation(self, input):
        N, C, H, W = input.shape
        stud_feats = self.student_branch(input)
        teacher_dict = self.teacher_branch(input, input)

        #(N,C,H,W), want norm along C axis
        # attns = []
        anomalies = []
        student_prototypes = []
        teacher_prototypes = []
        anom_res = torch.nn.Upsample(size=(H, W), mode='bilinear')
        big_anomalies = []
        for i in range(4):
            C = stud_feats[i].shape[1]

            student_terrain_prototype = stud_feats[i].mean(dim=(2,3), keepdim=True)
            teacher_aug_feats = teacher_dict["aug_feats"][i]
            a = student_terrain_prototype/torch.norm(student_terrain_prototype, dim=1, keepdim=True)
            b = teacher_aug_feats/torch.norm(teacher_aug_feats, dim=1, keepdim=True)
            anomaly_map = 1 - (a*b).sum(dim=1, keepdim=True)
            # anomaly_map = anomaly_map.repeat(1,C,1,1).detach() #detach anomaly map from gradients
            anomaly_map = self.abns[i](anomaly_map.detach())
            # anomaly_map = torch.zeros_like(anomaly_map.detach())
            anomalies.append(anomaly_map)
            big_anomalies.append(anom_res(anomaly_map))
        
        # seg_out = self.seg2(teacher_dict["aug_feats"][3], attns[3])
        # seg_out = self.seg3(seg_out, attns[2])
        # seg_out = self.seg4(seg_out, attns[1])
        # seg_out = self.seg5(seg_out, attns[0])
        # res = torch.nn.Upsample(size=tuple(aug_img.shape[-2:]), mode='bilinear')
        # seg_out = self.seg6(res(seg_out), aug_img)
        # seg_out = self.seg_outc(seg_out)


        resnet_out = self.deformation_encoder(input)
        c1 = self.activations['def_conv1']
        l1 = self.activations['def_layer1']
        l2 = self.activations['def_layer2']
        l3 = self.activations['def_layer3']
        l4 = self.activations['def_layer4']
        l1up = self.def_u2(torch.cat([l4, anomalies[-1]], dim=1), torch.cat([l3, anomalies[-2]], dim=1))
        l2up = self.def_u3(l1up, torch.cat([l2, anomalies[-3]], dim=1))
        l3up = self.def_u4(l2up, torch.cat([l1, anomalies[-4]], dim=1))
        l4up = self.def_u5(l3up, c1)
        l5up = self.def_u6(l4up, input)
        def_pred = self.def_outc(l5up)
        mag_pred = def_pred[:,0:10]
        angle_pred = def_pred[:,10:]
        anomaly_stack = torch.cat(big_anomalies, dim=1)
        payload = torch.cat([def_pred, anomaly_stack], dim=1)
        seg_pred = self.out_conv(payload)


        #(N,C,H,W), want norm along C axis
        # attns = []
        # student_prototypes = []
        # for i in range(4):
        #     C = stud_feats[i].shape[1]

        #     student_terrain_prototype = stud_feats[i].mean(dim=(2,3), keepdim=True)
        #     student_prototypes.append(student_terrain_prototype)

        #     teacher_aug_feats = teacher_dict["aug_feats"][i]
        #     a = student_terrain_prototype/torch.norm(student_terrain_prototype, dim=1, keepdim=True)
        #     b = teacher_aug_feats/torch.norm(teacher_aug_feats, dim=1, keepdim=True)
        #     anomaly_map = 1 - (a*b).sum(dim=1, keepdim=True)
        #     anomaly_map = anomaly_map.repeat(1,C,1,1).detach() #detach anomaly map from gradients
        #     f_map = torch.cat([def_feats[i], anomaly_map], dim=1)
        #     attn, _ = self.attn_layers[i](f_map)
        #     attns.append(attn)
        
        # seg_out = self.seg2(teacher_dict["aug_feats"][3], attns[3])
        # seg_out = self.seg3(seg_out, attns[2])
        # seg_out = self.seg4(seg_out, attns[1])
        # seg_out = self.seg5(seg_out, attns[0])
        # res = torch.nn.Upsample(size=tuple(input.shape[-2:]), mode='bilinear')
        # seg_out = self.seg6(res(seg_out), input)
        # seg_out = self.seg_outc(seg_out)


        



        # attn_map = nn.functional.softmax(torch.matmul(deformation_feats.permute(0,1,3,2), teacher_aug_feats), dim=2)
        # self_attn_map = self.self_attn_conv(torch.matmul(anomaly_map, attn_map))
        #1. deformation feats 
        #2. teacher aug_feats
        #3. anomaly map
        # resnet_out = self.resnet_encoder(x)
        # input = x
        # c1 = self.activations['conv1']
        # l1 = self.activations['layer1']
        # l2 = self.activations['layer2']
        # l3 = self.activations['layer3']
        # l4 = self.activations['layer4']
        # l1up = self.seg2(l4, l3)
        # l2up = self.seg3(l1up, l2)
        # l3up = self.seg4(l2up, l1)
        # l4up = self.seg5(l3up, c1)
        # l5up = self.seg6(l4up, input)
        # seg_pred = self.seg_outc(l5up)
        
        # up = torch.nn.Upsample(size=tuple(x.shape[-2:]), mode='bilinear')
        # l1_sim = (l3up*tl1/torch.norm(l3up, dim=1, keepdim=True)/torch.norm(tl1,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l2_sim = (l2up*tl2/torch.norm(l2up, dim=1, keepdim=True)/torch.norm(tl2,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l3_sim = (l1up*tl3/torch.norm(l1up, dim=1, keepdim=True)/torch.norm(tl3,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # all_feats = torch.cat([up(l1_sim), up(l2_sim), up(l3_sim)], dim=1).detach()
        # all_feats[torch.isnan(all_feats)] = 1e-9
        # rn2_out = self.decoder(all_feats)
        # seg_pred = self.seg_outc(rn2_out)
        
        # seg_pred = self.aspp(rn2_out)
        # if torch.any(torch.isnan(seg_pred)):
        #     print("oink")


        return seg_pred


    def get_patch_features(self, x):
        pred = self.resnet_encoder(x)
        l2 = self.activations["layer2"]
        l3 = self.activations["layer3"]
        return l2, l3

class DAAD(nn.Module):
    
    def __init__(self, classes, dims = [512,512,512,512,512,512,512,512,128]):
        super(DAAD, self).__init__()
        self.resnet_encoder = models.resnet34(pretrained=True)

        self.activations = {}
        self.resnet_encoder.conv1.register_forward_hook(self.get_activation('conv1'))
        self.resnet_encoder.layer1.register_forward_hook(self.get_activation('layer1'))
        self.resnet_encoder.layer2.register_forward_hook(self.get_activation('layer2'))
        self.resnet_encoder.layer3.register_forward_hook(self.get_activation('layer3'))
        self.resnet_encoder.layer4.register_forward_hook(self.get_activation('layer4'))

       
        # self.u1 = DoubleConv(512, 1024)
        self.u2 = Up(512, 256, 512)
        self.u3 = Up(256, 128, 256)
        self.u4 = Up(128, 64, 128)
        self.u5 = Up(64+32, 32, 64)
        self.u6 = Up(16+3, 16, 32)
        self.outc = OutputConv(16, classes)
        
        self.seg2 = Up(512, 256, 512)
        self.seg3 = Up(256, 128, 256)
        self.seg4 = Up(128, 64, 128)
        self.seg5 = Up(64+32, 32, 64)
        self.seg6 = Up(16+3, 16, 32)
        self.seg_outc = OutputConv(16, 1)

        embed_layers = []
        embed_layers.append(nn.Linear(1000, dims[0]))
        embed_layers.append(nn.BatchNorm1d(dims[0]))
        embed_layers.append(nn.ReLU(inplace=True))
        for d in dims[:-1]:
            embed_layers.append(nn.Linear(d, d))
            embed_layers.append(nn.BatchNorm1d(d))
            embed_layers.append(nn.ReLU(inplace=True))
        embed_layers.append(nn.Linear(dims[-2], dims[-1]))
        self.embedder = nn.Sequential(*embed_layers)
        self.domain_classifier = nn.Sequential(nn.Linear(dims[-1], dims[-1]), 
                                               nn.BatchNorm1d(dims[-1]), 
                                               nn.ReLU(inplace=True), 
                                             nn.Linear(dims[-1], 1))
        self.grl = GradientReversal()
        d_list = []
        for i in range(5):
            d_list.append(nn.Conv2d(3, 3, 3, padding=1))
            d_list.append(nn.BatchNorm2d(3))
            d_list.append(nn.ReLU())
            d_list.append(nn.Conv2d(3, 3, 3, padding=1))
            d_list.append(nn.BatchNorm2d(3))
        self.decoder = nn.Sequential(*d_list)
        # self.rn1 = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), 
        #                         nn.BatchNorm2d(3), 
        #                         nn.ReLU(),
        #                         nn.Conv2d(3, 3, 3, padding=1), 
        #                         nn.BatchNorm2d(3))
        # self.rn2 = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), 
        #                         nn.BatchNorm2d(3), 
        #                         nn.ReLU(),
        #                         nn.Conv2d(3, 3, 3, padding=1), 
        #                         nn.BatchNorm2d(3))
        self.aspp = ASPP(3, [2, 4], 1)

    

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook
    

    def forward(self, x):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        x = self.u1(l4)
        x = self.u2(x, l3)
        x = self.u3(x, l2)
        x = self.u4(x, l1)
        x = self.u5(x, c1)
        x = self.u6(x, input)
        seg_out = self.outc(x)
        
        embedding = self.embedder(resnet_out)
        reverse_embed = self.grl(embedding)
        domain_out = self.domain_classifier(reverse_embed)

        return seg_out, domain_out, embedding

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            try:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    
                    param = param.data
                own_state[name].copy_(param)
                print("loaded success:", name)
            except:
                print("failed")
                pass

    def denoise_and_feature(self, x):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        # l1up = self.u1(l4)
        l1up = self.u2(l4, l3)
        l2up = self.u3(l1up, l2)
        l3up = self.u4(l2up, l1)
        l4up = self.u5(l3up, c1)
        l5up = self.u6(l4up, input)
        seg_out = self.outc(l5up)
        
        # embedding = self.embedder(resnet_out)
        # reverse_embed = self.grl(embedding)
        # domain_out = self.domain_classifier(reverse_embed)
        # res = torch.nn.Upsample(size=tuple(l1.shape[-2:]), mode='bilinear')
        # l3_resize = torch.nn.Upsample(l3, size=tuple(l1.shape[-2:]), mode='bilinear')
        # l2up_resize = res(l2up)
        # l3up_resize = res(l3up)
        # l4up_resize = res(l4up)
        return seg_out, l1, l2, l3
    
    def denoise_and_seg(self, x):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        # l1up = self.u1(l4)
        dl1up = self.u2(l4, l3)
        dl2up = self.u3(dl1up, l2)
        dl3up = self.u4(dl2up, l1)
        dl4up = self.u5(dl3up, c1)
        dl5up = self.u6(dl4up, input)
        denoise_out = self.outc(dl5up)
        
        sl1up = self.seg2(l4, l3)
        sl2up = self.seg3(sl1up, l2)
        sl3up = self.seg4(sl2up, l1)
        sl4up = self.seg5(sl3up, c1)
        sl5up = self.seg6(sl4up, input)
        seg_pred = self.seg_outc(sl5up)
        
        # embedding = self.embedder(resnet_out)
        # reverse_embed = self.grl(embedding)
        # domain_out = self.domain_classifier(reverse_embed)
        # res = torch.nn.Upsample(size=tuple(l1.shape[-2:]), mode='bilinear')
        # l3_resize = torch.nn.Upsample(l3, size=tuple(l1.shape[-2:]), mode='bilinear')
        # l2up_resize = res(l2up)
        # l3up_resize = res(l3up)
        # l4up_resize = res(l4up)
        return denoise_out, seg_pred

    def predict_segmentation(self, x, tl1, tl2, tl3):
        resnet_out = self.resnet_encoder(x)
        input = x
        c1 = self.activations['conv1']
        l1 = self.activations['layer1']
        l2 = self.activations['layer2']
        l3 = self.activations['layer3']
        l4 = self.activations['layer4']
        l1up = self.seg2(l4, l3)
        l2up = self.seg3(l1up, l2)
        l3up = self.seg4(l2up, l1)
        l4up = self.seg5(l3up, c1)
        l5up = self.seg6(l4up, input)
        seg_pred = self.seg_outc(l5up)
        
        # up = torch.nn.Upsample(size=tuple(x.shape[-2:]), mode='bilinear')
        # l1_sim = (l3up*tl1/torch.norm(l3up, dim=1, keepdim=True)/torch.norm(tl1,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l2_sim = (l2up*tl2/torch.norm(l2up, dim=1, keepdim=True)/torch.norm(tl2,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # l3_sim = (l1up*tl3/torch.norm(l1up, dim=1, keepdim=True)/torch.norm(tl3,dim=1, keepdim=True)).sum(dim=1, keepdim=True)
        # all_feats = torch.cat([up(l1_sim), up(l2_sim), up(l3_sim)], dim=1).detach()
        # all_feats[torch.isnan(all_feats)] = 1e-9
        # rn2_out = self.decoder(all_feats)
        # seg_pred = self.seg_outc(rn2_out)
        
        # seg_pred = self.aspp(rn2_out)
        if torch.any(torch.isnan(seg_pred)):
            print("oink")


        return seg_pred
        


    def get_patch_features(self, x):
        pred = self.resnet_encoder(x)
        l2 = self.activations["layer2"]
        l3 = self.activations["layer3"]
        return l2, l3






# BASNET

## code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torchvision
from torchvision import models
import torch.nn.functional as F

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'ResNet34P','ResNet50S','ResNet50P','ResNet101P']
#
# resnet18_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet18-5c106cde.pth'
# resnet34_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet34-333f7ec4.pth'
# resnet50_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet50-19c8e357.pth'
# resnet101_dir = '/local/sda4/yqian3/RoadNets/resnet_model/resnet101-5d3b4d8f.pth'
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDe(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockDe, self).__init__()

        self.convRes = conv3x3(inplanes,planes,stride)
        self.bnRes = nn.BatchNorm2d(planes)
        self.reluRes = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.convRes(x)
        residual = self.bnRes(residual)
        residual = self.reluRes(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    





class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class BASNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(BASNet,self).__init__()

        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #224
        #stage 2
        self.encoder2 = resnet.layer2 #112
        #stage 3
        self.encoder3 = resnet.layer3 #56
        #stage 4
        self.encoder4 = resnet.layer4 #28

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #14

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #7

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)


    def forward(self,x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # 256
        h2 = self.encoder2(h1) # 128
        h3 = self.encoder3(h2) # 64
        h4 = self.encoder4(h3) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256

        ## -------------Refine Module-------------
        dout = self.refunet(d1) # 256

        return F.sigmoid(dout), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(db)
    

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

