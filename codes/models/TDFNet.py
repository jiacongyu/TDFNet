from models.rcrnet_vit import RCRNet_vit
from models.rcrnet_res2 import RCRNet_res2net
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import cv2
from models.Equirec2Cube import Equirec2Cube
from models.Cube2Equirec import Cube2Equirec
from spherical_pos_emb import SphericalPositionEmbedding
from models.SphericalTangentBranch import SphericalTangentBranch
from models.panorama_lgf import LGF

num_cube = 6


class ImgModel(nn.Module):
    def __init__(self, output_stride=16, fusion_strategy='geometry_only'):
        super(ImgModel, self).__init__()
        self.backbone = RCRNet_vit(
            n_classes=1,
            output_stride=output_stride,
        )

        self.backbone_cube = RCRNet_res2net(
            n_classes=1,
            output_stride=output_stride,
            pretrained=True
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tangent_branch = SphericalTangentBranch(
            erp_size=(640, 1280),
            nrows=[4],
            patch_sizes=[224],
            fovs=[(80, 80)],
            shifts=[False],
            device=device,
            emb_dim=256
        )

        ER_height = 640
        CB_height = ER_height // 2
        ER_m_height = 16

        self.equi2cube = Equirec2Cube(1, ER_height, ER_height * 2, CB_height, 90)

        ER_m_height = 20
        ER_m_width = 40
        self.cube2equi = Cube2Equirec(1, ER_m_height, ER_m_width, 90)
        self.equi2cube_aux = Equirec2Cube(num_cube, ER_m_height, ER_m_width, 20, 90)

        self.shapeCube = CB_height

        self.fusion_strategy = fusion_strategy
        self.fusion_module = LGF(
            in_channel=256,
            strategy=fusion_strategy
        )

        self.spherical_pos_emb = SphericalPositionEmbedding(
            emb_dim=256, erp_size=(640, 1280), cmp_face_size=320, num_tangent_patches=18
        )

        if self.training:
            self.initialize_w_image_pretrain()

            # ------------------------------------------------------------------

    def initialize_w_image_pretrain(self):
        backbone_vit_pretrain = torch.load(
             r'D:\\study\\qjxzxjc\\TDFNet\\TDFNet_models\\pretrain\\RCRNet_vit_pretrain.pth',
            map_location='cpu')
        backbone_res2net_pretrain = torch.load(
            r'D:\\study\\qjxzxjc\\TDFNet\\TDFNet_models\\pretrain\\RCRNet_res2_pretrain.pth',
            map_location='cpu')

        all_params = {}
        for k, v in self.backbone.state_dict().items():
            if k in backbone_vit_pretrain.keys():
                v = backbone_vit_pretrain[k]
                all_params[k] = v
        self.backbone.load_state_dict(all_params, strict=False)

        all_params_cube = {}
        for k, v in self.backbone_cube.state_dict().items():
            if k in backbone_res2net_pretrain.keys():
                v = backbone_res2net_pretrain[k]
                all_params_cube[k] = v
        self.backbone_cube.load_state_dict(all_params_cube, strict=False)

        # ------------------------------------------------------------------

    def forward(self, frame, ER_frame, ER_gt=None):
        feats = self.backbone.feat_conv(frame)
        cubes_in = self.equi2cube.ToCubeTensor(ER_frame)
        cube_feats = self.backbone_cube.feat_conv(cubes_in)
        cube_feat_high = cube_feats[3]
        cubes_bottleneck = self.cube2equi.ToEquirecTensor(cube_feat_high)
        tangent_feat = self.tangent_branch(ER_frame, scale=0)

        erp_bottleneck = feats[3]  # [B, 256, 20, 40]

        enhanced_erp, enhanced_cmp, enhanced_tangent = self.spherical_pos_emb(
            erp_bottleneck, cubes_bottleneck, tangent_feat
        )

        _, _, h_erp, w_erp = enhanced_erp.shape
        if enhanced_cmp.shape[-2:] != (h_erp, w_erp):
            enhanced_cmp = F.interpolate(
                enhanced_cmp, size=(h_erp, w_erp), mode='bilinear', align_corners=False
            )

        B, C, H, W = enhanced_erp.shape

        y_coords = torch.linspace(-1.0, 1.0, steps=H, device=enhanced_erp.device)
        lat_map = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)

        refined_fused = self.fusion_module(enhanced_erp, enhanced_cmp, lat_map, enhanced_tangent)

        h_target, w_target = feats[2].shape[-2:]
        feats_bottleneck = F.interpolate(refined_fused, size=(h_target, w_target), mode='bilinear', align_corners=False)

        if ER_gt is None:
            pred = self.backbone.seg_conv(feats[0], feats[1], feats[2], feats_bottleneck, frame.shape[2:])
            return pred
        else:
            pred = self.backbone.seg_conv(feats[0], feats[1], feats[2], feats_bottleneck, frame.shape[2:])

            feats_aux = enhanced_cmp  # [B, 256, 20, 40]

            feats_aux = F.interpolate(feats_aux, size=(20, 40), mode='bilinear', align_corners=False)

            feats_aux_cube = self.equi2cube_aux.ToCubeTensor(feats_aux)

            feats_cube_bottleneck = cube_feats[3] + feats_aux_cube

            pred_aux = self.backbone_cube.seg_conv(
                cube_feats[0], cube_feats[1], cube_feats[2], feats_cube_bottleneck,
                [self.shapeCube, self.shapeCube]
            )

            cube_gt = self.equi2cube.ToCubeTensor(ER_gt)

            return pred, pred_aux, cube_gt
