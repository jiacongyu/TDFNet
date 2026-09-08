from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_dilation import Bottleneck, conv1x1
from models.blocks import (
    _make_encoder,
    forward_vit,
)
from models.cda import CDA_PixelDecoder


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class _RefinementModule(nn.Module):
    """ Reduce channels and refinment module"""
    def __init__(self,
        bottom_up_channels,
        reduce_channels,
        top_down_channels,
        refinement_channels,
        expansion=2
    ):
        super(_RefinementModule, self).__init__()
        downsample = None
        if bottom_up_channels != reduce_channels:
            downsample = nn.Sequential(
                conv1x1(bottom_up_channels, reduce_channels),
                nn.BatchNorm2d(reduce_channels),
            )
        self.skip = Bottleneck(bottom_up_channels, reduce_channels // expansion, 1, 1, downsample, expansion)
        self.refine = _ConvBatchNormReLU(reduce_channels + top_down_channels, refinement_channels, 3, 1, 1, 1)

    def forward(self, td, bu):
        td = self.skip(td)  # [B, reduce_channels, H, W]
        x = torch.cat((bu, td), dim=1)
        x = self.refine(x)

        return x


class RCRNet_vit(nn.Module):
    def __init__(self, n_classes, output_stride):
        super(RCRNet_vit, self).__init__()

        # vit encoder
        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }
        self.pretrained = _make_encoder(
            "vitb_rn50_384",
            256,
            False,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks["vitb_rn50_384"],
            use_readout="project",
            enable_attention_hooks=False,
        )

        self.cda = CDA_PixelDecoder(
            in_channels_list=[256, 512, 768, 768],
            conv_dim=256,
            mask_dim=256,
            num_layers=3,
            nheads=8,
            fpn_levels=3,
        )

        # Decoder
        self.decoder = nn.Sequential(
                    OrderedDict(
                        [
                            ("conv1", _ConvBatchNormReLU(128, 256, 3, 1, 1, 1)),
                            ("conv2", nn.Conv2d(256, n_classes, kernel_size=1)),
                        ]
                    )
                )
        self.add_module("refinement1", _RefinementModule(768, 96, 256, 128, 2))
        self.add_module("refinement2", _RefinementModule(512, 96, 128, 128, 2))
        self.add_module("refinement3", _RefinementModule(256, 96, 128, 128, 2))

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def feat_conv(self, x):

        block1, block2, block3, block4 = forward_vit(self.pretrained, x)

        multi_feats = {
            "res2": block1,
            "res3": block2,
            "res4": block3,
            "res5": block4
        }
        mask_feat, multi_scale_feats = self.cda(multi_feats, masks=None)

        mask_feat = multi_scale_feats[0]

        return block1, block2, block3, mask_feat

    def seg_conv(self, block1, block2, block3, block4, shape):

        bu1 = self.refinement1(block3, block4)
        bu1 = F.interpolate(bu1, size=block2.shape[2:], mode="bilinear", align_corners=False)
        bu2 = self.refinement2(block2, bu1)
        bu2 = F.interpolate(bu2, size=block1.shape[2:], mode="bilinear", align_corners=False)
        bu3 = self.refinement3(block1, bu2)
        bu3 = F.interpolate(bu3, size=shape, mode="bilinear", align_corners=False)
        seg = self.decoder(bu3)

        return seg

    def forward(self, x):
        block1, block2, block3, block4 = self.feat_conv(x)
        seg = self.seg_conv(block1, block2, block3, block4, x.shape[2:])

        return seg



if __name__ == "__main__":
    model = RCRNet_vit(n_classes=1, output_stride=16)
    x = torch.randn(6, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        feats = model.feat_conv(x)
        print("🔍 feat_conv outputs:", [f.shape for f in feats])
        out = model(x)
    print(f"✅ CDA替换成功！输出形状: {out.shape}")