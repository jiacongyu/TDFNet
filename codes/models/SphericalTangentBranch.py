import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from einops import rearrange  # pip install einops

PI = np.pi
PI_2 = np.pi / 2

def pair(t):
    return t if isinstance(t, (tuple, list)) else (t, t)

def uv2xyz(uv):
    u, v = uv[..., 0], uv[..., 1]
    x = torch.cos(v) * torch.cos(u)
    y = torch.cos(v) * torch.sin(u)
    z = torch.sin(v)
    return torch.stack([x, y, z], -1)

class Equi2Pers(nn.Module):

    def __init__(self, erp_size=(256, 512), fov=(80, 80), nrows=4, patch_size=224, shift=False, geom=None, device='cuda'):
        super(Equi2Pers, self).__init__()
        assert geom in [None, 'cubemap', 'icosahedron']
        self.erp_size = pair(erp_size)
        self.fov = pair(fov)
        self.nrows = nrows
        self.patch_size = pair(patch_size)
        self.shift = shift
        self.geom = geom
        self.device = device

        height, width = self.patch_size
        erp_h, erp_w = self.erp_size
        FOV = torch.tensor(self.fov, dtype=torch.float32).to(device) / 180 * PI

        use_cubemap = (geom == 'cubemap')
        if use_cubemap:
            num_cols = [1, 4, 1]
            phi_centers = np.array([-90, 0, 90], dtype=np.float32)
        else:
            if nrows == 3:
                num_cols = [3, 4, 3]
                phi_centers = np.array([-60, 0, 60], dtype=np.float32)
            elif nrows == 4:
                num_cols = [3, 6, 6, 3]
                phi_centers = np.array([-72, -24, 24, 72], dtype=np.float32)
            elif nrows == 5:
                num_cols = [3, 6, 8, 6, 3]
                phi_centers = np.array([-80, -48, -16, 16, 80], dtype=np.float32)
            elif nrows == 6:
                num_cols = [3, 6, 8, 8, 6, 3]
                phi_centers = np.array([-85, -60, -35, 0, 35, 85], dtype=np.float32)
            else:
                raise ValueError("nrows must be 3,4,5 or 6")

        all_combos = []
        num_rows = len(num_cols)
        for i in range(num_rows):
            for j in range(num_cols[i]):
                theta_interval = 360 / num_cols[i]
                if self.shift:
                    theta_center = (j + 0.5) * theta_interval - 180
                else:
                    theta_center = j * theta_interval - 180 + theta_interval / 2
                center = [theta_center, phi_centers[i]]
                all_combos.append(center)
        all_combos = np.vstack(all_combos)
        num_patch = all_combos.shape[0]
        self.num_patch = num_patch

        center_point = torch.from_numpy(all_combos).float().to(device)
        center_point[:, 0] = (center_point[:, 0] + 180) / 360
        center_point[:, 1] = (center_point[:, 1] + 90) / 180
        cp = center_point * 2 - 1
        self.center_p = cp.clone()
        cp_rad = cp.clone()
        cp_rad[:, 0] *= PI
        cp_rad[:, 1] *= PI_2
        cp_rad = cp_rad.unsqueeze(1).repeat(1, height * width, 1)

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device)
        )
        screen_points = torch.stack([xx.flatten(), 1 - yy.flatten()], -1).float()
        convertedCoord = screen_points * 2 - 1
        convertedCoord[:, 0] *= PI
        convertedCoord[:, 1] *= PI_2
        convertedCoord *= FOV[None, :]
        convertedCoord = convertedCoord.unsqueeze(0).repeat(num_patch, 1, 1)

        x, y = convertedCoord[:, :, 0], convertedCoord[:, :, 1]
        rou = torch.sqrt(x**2 + y**2 + 1e-8)
        c = torch.atan(rou)
        sin_c, cos_c = torch.sin(c), torch.cos(c)
        lat = torch.asin(cos_c * torch.sin(cp_rad[:, :, 1]) + y * sin_c * torch.cos(cp_rad[:, :, 1]) / rou)
        lon = cp_rad[:, :, 0] + torch.atan2(x * sin_c, rou * cos_c * torch.cos(cp_rad[:, :, 1]) - y * sin_c * torch.sin(cp_rad[:, :, 1]))

        lat_new = lat / PI_2
        lon_new = lon / PI
        lon_new = torch.clamp(lon_new, -1, 1)
        lon_new[lon_new > 1] -= 2
        lon_new[lon_new < -1] += 2

        lon_new = lon_new.view(num_patch, height, width).permute(1, 0, 2).contiguous().view(height, num_patch * width)
        lat_new = lat_new.view(num_patch, height, width).permute(1, 0, 2).contiguous().view(height, num_patch * width)
        self.grid = torch.stack([lon_new, lat_new], -1).to(device)

        grid_tmp = torch.stack([lon, lat], -1)
        xyz = uv2xyz(grid_tmp)
        xyz = xyz.view(num_patch, height, width, 3).permute(0, 3, 1, 2)
        self.xyz = xyz.to(device)

    def project(self, erp_img):
        bs, c, erp_h, erp_w = erp_img.shape
        height, width = self.patch_size
        num_patch = self.num_patch

        grid = self.grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)
        pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        pers = pers.view(bs, c, height, num_patch * width)
        patches = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
        patches = patches.transpose(1, 2).contiguous().view(bs, num_patch, c, height, width)  # 修复：transpose 后 contiguous + view
        return patches

class TangentFeatureExtractor(nn.Module):

    def __init__(self, patch_size=224, device='cuda', emb_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.device = device

        resnet = models.resnet18(pretrained=True)
        resnet.eval()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # [512,7,7]

        self.down = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if emb_dim != 512:
            self.proj = nn.Linear(512, emb_dim)  # [bs*num_patch, 512] → [bs*num_patch, emb_dim]
            nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')  # 初始化
        else:
            self.proj = None

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.to(device)

    def forward(self, tangent_imgs, num_patch=None):

        bs, t, c, h, w = tangent_imgs.shape
        num_patch = t if num_patch is None else num_patch
        assert c == 3 and h == w == self.patch_size

        tangent_imgs = tangent_imgs.contiguous()
        flat_imgs = tangent_imgs.view(bs * num_patch, c, h, w).to(self.device)

        resnet_features = self.feature_extractor(flat_imgs)  # [bs*num_patch, 512, 7, 7]
        features_pooled = self.down(resnet_features)  # [bs*num_patch, 512, 1, 1]

        features_flat = features_pooled.squeeze(-1).squeeze(-1)  # [bs*num_patch, 512]
        if self.proj is not None:
            features_flat = self.proj(features_flat)  # [bs*num_patch, emb_dim=256]

        features = features_flat.view(bs, num_patch, self.emb_dim)

        return features


class SphericalTangentBranch(nn.Module):
    def __init__(self, erp_size=(960, 1920), nrows=[3, 4], patch_sizes=[224, 224], fovs=[(80,80), (80,80)],
                 shifts=[False, True], device='cuda', emb_dim=512):
        super().__init__()
        self.device = device
        self.num_scales = len(nrows)
        self.n_patches = []
        self.patch_sizes = patch_sizes

        self.E2P = nn.ModuleList()
        for i in range(self.num_scales):
            tangent_conf = {'nrows': nrows[i], 'patch_size': patch_sizes[i], 'fov': fovs[i]}
            shift_ = shifts[i]
            projector = Equi2Pers(erp_size=erp_size, shift=shift_, device=device, **tangent_conf)
            self.E2P.append(projector)
            self.n_patches.append(projector.num_patch)

        max_patch = max(self.n_patches)
        first_patch_size = patch_sizes[0]
        self.feature_extractor = TangentFeatureExtractor(patch_size=first_patch_size, device=device, emb_dim=emb_dim)

    def forward(self, erp_img, scale=0):
        assert scale < self.num_scales

        projector = self.E2P[scale]
        num_patch = projector.num_patch
        curr_patch_size = self.patch_sizes[scale]

        x_tang = projector.project(erp_img)  # [BS, num_patch, 3, H, W]

        if curr_patch_size != self.feature_extractor.patch_size:
            bs = x_tang.shape[0]
            x_tang = F.adaptive_avg_pool2d(
                x_tang.reshape(bs * num_patch, 3, curr_patch_size, curr_patch_size),
                (224, 224)
            ).reshape(bs, num_patch, 3, 224, 224)

        with torch.no_grad():
            features = self.feature_extractor(x_tang, num_patch=num_patch)  # [BS, num_patch, 512]

        return features
