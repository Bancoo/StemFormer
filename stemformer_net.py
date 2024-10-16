import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilevit import mobile_vit_xx_small as mobilevit
from nets.mobilevit import ConvLayer
from nets.mobilevit_config import get_config
from typing import Optional, Tuple, Union, Dict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)

		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result
     
class StripPooling(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
    
class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out

class Stemformer(nn.Module):
    def __init__(self, num_classes, backbone="mobilevit", pretrained=True, downsample_factor=16):
        super(Stemformer, self).__init__()

        if backbone == "mobilevit":
            backbone = mobilevit()
            in_channels = 80

            self.conv_1 = backbone.conv_1
            self.layer_1 = backbone.layer_1
            self.layer_2 = backbone.layer_2
            self.layer_3 = backbone.layer_3
            self.layer_4 = backbone.layer_4
            self.layer_5 = backbone.layer_5

            self.layer_33_ = backbone.layer_33_
            self.layer_34_ = backbone.layer_34_
            self.layer_35_ = backbone.layer_35_

            self.conv_1x1_exp = backbone.conv_1x1_exp
                
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilevit.'.format(backbone))

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        self.mpm = StripPooling(in_channels=80, pool_size=(16, 8), norm_layer=nn.BatchNorm2d,up_kwargs=up_kwargs)

        self.cat_conv = nn.Sequential(
            nn.Conv2d(24+256, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        
        self.relu = nn.ReLU(inplace=False)
        self.cls_conv = nn.Conv2d(64, num_classes, 1, stride=1)
        planes=32
        spp_planes=64
        head_planes=128
        bn_mom = 0.1
        highres_planes = planes * 2

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(64, 24, kernel_size=1, bias=False),
                                          BatchNorm2d(24, momentum=bn_mom),
                                          )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(24, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(64, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(64, momentum=bn_mom),
                                   )
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        width_output = x.shape[-1] // 4  # 128 * 128
        height_output = x.shape[-2] // 4
        layers = []

        # print("input x      ",x.size())
        # stage 1
        x = self.conv_1(x)
        x = self.layer_1(x)
        layers.append(x)
        
        # stage 2
        x = self.layer_2(self.relu(x))
        layers.append(x)  # 1 - B

        # 分支
        # stage 6
        x = self.layer_3(self.relu(x))
        layers.append(x)  # 2 - C

        # stage 3
        x_ = self.layer_33_(self.relu(layers[1]))
        
        # stage 7
        x = self.layer_4(self.relu(x))
        layers.append(x)    # 3-D 

        # stage 4
        x_ = self.layer_34_(self.relu(x_))
        # print("block B''  2 ",x_.size())
        
        # BFF,down
        x = x + self.down4(self.relu(x_))

        # BFF,up
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
        # print("block B''' 1 ",x_.size())
        
        # stage 5
        x_ = self.layer_34_(self.relu(x_)) 

        # layer5 是 stage 8
        x = F.interpolate(
                        self.aspp(self.mpm(self.layer_5(self.relu(x)))),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
        
        # seghead
        x = self.cat_conv(torch.cat((x, x_), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x

