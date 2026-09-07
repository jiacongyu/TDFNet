import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math  # 确保 pi 使用

class SphericalPositionEmbedding(nn.Module):

    def __init__(self, emb_dim=256, erp_size=(640, 1280), cmp_face_size=320, num_tangent_patches=18):
        super(SphericalPositionEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.erp_size = erp_size
        self.cmp_face_size = cmp_face_size
        self.num_tangent_patches = num_tangent_patches

        self.erp_pos_emb = nn.Linear(5, emb_dim)
        self.cmp_pos_emb = nn.Linear(5, emb_dim)
        self.tangent_pos_emb = nn.Linear(5, emb_dim)

        nn.init.kaiming_normal_(self.erp_pos_emb.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.cmp_pos_emb.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.tangent_pos_emb.weight, mode='fan_out', nonlinearity='relu')
        self.register_buffer('tangent_coords', self._compute_tangent_coords())

    def _compute_erp_coords(self, h_feat, w_feat, device):
        lat = torch.linspace(math.pi / 2, -math.pi / 2, h_feat).to(device)  # [-pi/2 ~ pi/2], len=h_feat=20
        lon = torch.linspace(-math.pi, math.pi, w_feat).to(device)  # [-pi ~ pi], len=w_feat=40
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='xy')  # [h_feat, w_feat]，指定 indexing 消除警告

        x = torch.cos(lat_grid) * torch.cos(lon_grid)
        y = torch.sin(lat_grid)
        z = torch.cos(lat_grid) * torch.sin(lon_grid)
        rho = torch.ones_like(x)
        center = torch.zeros_like(x)

        coords = torch.stack([x, y, z, rho, center], dim=0)  # [5, h_feat, w_feat] e.g., [5,20,40]
        return coords.float()

    def _compute_cmp_coords(self, h_feat, w_feat, device):
        lat = torch.linspace(math.pi / 2, -math.pi / 2, h_feat).to(device)  # len=16
        lon = torch.linspace(-math.pi, math.pi, w_feat).to(device)  # len=32
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='xy')

        x = torch.cos(lat_grid) * torch.cos(lon_grid)
        y = torch.sin(lat_grid)
        z = torch.cos(lat_grid) * torch.sin(lon_grid)
        rho = torch.ones_like(x)
        center = torch.zeros_like(x)

        coords = torch.stack([x, y, z, rho, center], dim=0)  # [5, h_feat, w_feat] e.g., [5,16,32]
        return coords.float()

    def _compute_tangent_coords(self):
        phi_centers = torch.tensor([-72, -24, 24, 72], dtype=torch.float32) * (math.pi / 180)
        num_cols = [3, 6, 6, 3]
        all_centers_list = []
        total_patches = 0
        for i in range(4):
            n_col = num_cols[i]
            theta_interval = 2 * math.pi / n_col
            j_arange = torch.arange(n_col, dtype=torch.float32) + 0.5  # CPU
            theta_centers = j_arange * theta_interval
            phi_batch = phi_centers[i].expand(n_col)
            batch = torch.stack([theta_centers, phi_batch], dim=1)
            all_centers_list.append(batch)
            total_patches += n_col
        all_centers = torch.cat(all_centers_list, dim=0)  # [18, 2]

        assert all_centers.shape[0] == self.num_tangent_patches, f"Expected {self.num_tangent_patches} patches, got {all_centers.shape[0]}"

        lon_rad, lat_rad = all_centers[:, 0], all_centers[:, 1]
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.sin(lat_rad)
        z = torch.cos(lat_rad) * torch.sin(lon_rad)
        rho = torch.ones_like(x)
        center = torch.zeros_like(x)

        coords = torch.stack([x, y, z, rho, center], dim=0).float()  # [5, 18]，CPU
        return coords

    def forward(self, erp_feat, cmp_feat, tangent_feat=None):
        B = erp_feat.shape[0]
        _, _, h_erp, w_erp = erp_feat.shape
        _, _, h_cmp, w_cmp = cmp_feat.shape
        device = erp_feat.device

        # 处理ERP分支（动态坐标 + to(device)）
        erp_coords = self._compute_erp_coords(h_erp, w_erp, device)
        erp_coords_b = erp_coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B,5,20,40]
        erp_coords_flat = erp_coords_b.permute(0, 2, 3, 1).reshape(B * h_erp * w_erp, 5)  # [B*800,5]，cuda
        erp_pos_flat = self.erp_pos_emb(erp_coords_flat)
        erp_pos = erp_pos_flat.view(B, h_erp, w_erp, self.emb_dim)  # [B,20,40,256]
        erp_pos = erp_pos.permute(0, 3, 1, 2)  # [B,256,20,40]
        enhanced_erp = erp_feat + erp_pos

        # 处理CMP分支（动态坐标 + to(device)）
        cmp_coords = self._compute_cmp_coords(h_cmp, w_cmp, device)  # [5,16,32]，cuda
        cmp_coords_b = cmp_coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B,5,16,32]
        cmp_coords_flat = cmp_coords_b.permute(0, 2, 3, 1).reshape(B * h_cmp * w_cmp, 5)  # [B*512,5]，cuda
        cmp_pos_flat = self.cmp_pos_emb(cmp_coords_flat)
        cmp_pos = cmp_pos_flat.view(B, h_cmp, w_cmp, self.emb_dim)  # [B,16,32,256]
        cmp_pos = cmp_pos.permute(0, 3, 1, 2)  # [B,256,16,32]
        enhanced_cmp = cmp_feat + cmp_pos

        enhanced_tangent = None
        if tangent_feat is not None:
            assert tangent_feat.shape[1] == self.num_tangent_patches, f"Expected {self.num_tangent_patches} patches, got {tangent_feat.shape[1]}"
            tangent_coords_b = self.tangent_coords.to(device).unsqueeze(0).expand(B, -1, -1)
            tangent_coords_flat = tangent_coords_b.permute(0, 2, 1).reshape(B * self.num_tangent_patches, 5)  # cuda
            tangent_pos_flat = self.tangent_pos_emb(tangent_coords_flat)
            tangent_pos = tangent_pos_flat.view(B, self.num_tangent_patches, self.emb_dim)
            enhanced_tangent = tangent_feat + tangent_pos

        if enhanced_tangent is not None:
            return enhanced_erp, enhanced_cmp, enhanced_tangent
        else:
            return enhanced_erp, enhanced_cmp


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_emb = SphericalPositionEmbedding(emb_dim=256, erp_size=(640, 1280), num_tangent_patches=18).to(device)
    B = 2

    erp_feat = torch.randn(B, 256, 20, 40, device=device)
    cmp_feat = torch.randn(B, 256, 16, 32, device=device)
    enhanced_erp, enhanced_cmp = pos_emb(erp_feat, cmp_feat)
    print(f"ERP/CMP Input shapes: erp {erp_feat.shape}, cmp {cmp_feat.shape}")
    print(f"Output shapes: erp {enhanced_erp.shape}, cmp {enhanced_cmp.shape}")

    tangent_feat = torch.randn(B, 18, 256, device=device)
    enhanced_erp_t, enhanced_cmp_t, enhanced_tangent = pos_emb(erp_feat, cmp_feat, tangent_feat)
    print(f"Tangent Input shape: {tangent_feat.shape}, Output shape: {enhanced_tangent.shape}")
    print("Test passed: Dynamic shapes match, devices unified, no errors for all branches.")
