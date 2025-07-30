import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(ch, ch // r, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch // r, ch, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w
    
class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        stacked = torch.cat([avg_out, max_out], dim=1)
        attention_map = torch.sigmoid(self.bn(self.conv(stacked)))
        return x * attention_map
    
class ResidualCBAMSEBlock(nn.Module):
    def __init__(self, ch, dilation=1, drop_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        
        self.se = SE(ch) # 
        self.spatial_attn = CBAMSpatialAttention(kernel_size=7) # 空间注意力
            
        self.drop  = DropBlock2d(block_size=7, drop_prob=drop_p) if drop_p > 0 else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.se(out)
        out = self.spatial_attn(out)
        
        out = self.drop(out)
        return self.relu(out + identity)
    
class EnergyAwareCoordResConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, energy_ch: int = 2, k: int = 3, bias: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.energy_ch = energy_ch
        
        total_in_ch = in_ch + 2 + energy_ch
        self.main_conv = nn.Conv2d(total_in_ch, out_ch, kernel_size=k,
                                padding=k // 2, bias=bias)
        
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) \
                    if in_ch != out_ch else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, energy):
        b, _, h, w = x.shape
        
        assert energy.shape[0] == b and energy.shape[2] == h and energy.shape[3] == w, \
            "Energy tensor shape mismatch!"
        assert energy.shape[1] == self.energy_ch, \
            f"Expected energy channels {self.energy_ch}, but got {energy.shape[1]}"

        yy, xx = torch.meshgrid(
            torch.linspace(-1., 1., h, device=x.device, dtype=x.dtype),
            torch.linspace(-1., 1., w, device=x.device),
            indexing='ij'
        )
        coords = torch.stack([xx, yy], dim=0)      # 2×H×W
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1)  # B×2×H×W

        combined_input = torch.cat([x, coords, energy], dim=1)
        
        y = self.main_conv(combined_input)
        
        res = self.proj(x)
        
        return self.relu(y + res)



class DropBlock2d(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()
        mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        return x * (1.0 - mask) * (mask.numel() / mask.sum().clamp(min=1.))


class GatedFusionModule(nn.Module):
    def __init__(self, internal_channels: int, external_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(internal_channels + external_channels, internal_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_channels // 2, external_channels, 1),
            nn.Sigmoid()
        )
        self.projector = nn.Conv2d(external_channels, internal_channels, kernel_size=1)
        
    def forward(self, internal_feat, external_feat):
        combined = torch.cat([internal_feat, external_feat], dim=1)
        gate = self.gate_conv(combined)
        weighted_external_feat = external_feat * gate
        projected_external_feat = self.projector(weighted_external_feat)
        return internal_feat + projected_external_feat

class RFMfold(nn.Module):
    def __init__(self,
        drop_p: float = 0.1,
        main_ch: int = 16,       
        energy_ch: int = 2,
        ss_fea_ch: int = 6,      
        base_ch: int = 128,
        depth: int = 6,           
        dilations=(1, 2, 4, 8, 16)):
        super().__init__()  

        self.main_stem = EnergyAwareCoordResConv2d(main_ch+1, base_ch, energy_ch, k=3, bias=False)
    
        self.prior_stem = nn.Sequential(
            nn.Conv2d(ss_fea_ch + 1, base_ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch, kernel_size=3, padding=1),
        )

        blocks = []
        for i in range(depth):
            blk = ResidualCBAMSEBlock( 
                ch=base_ch, 
                dilation=dilations[i % len(dilations)],
                drop_p=drop_p if i >= depth // 2 else 0.0
            )
            blocks.append(blk)
        self.trunk = nn.Sequential(*blocks)
    
        self.internal_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.fusion_module = GatedFusionModule(
            internal_channels=base_ch,
            external_channels=ss_fea_ch + 1 # ss_fea(5) + mask(1) = 6
        )
    
        self.post_fusion_processor = nn.Sequential(
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)   
        )

        self.final_head = nn.Conv2d(base_ch, 1, 1)
    
    def forward(self, x_outer, energy, ss_fea, mask):
        x_main_masked = torch.cat([x_outer, mask], dim=1)
        main_feat = self.main_stem(x_main_masked, energy)
        ss_fea_masked = torch.cat([ss_fea, mask], dim=1)
        prior_feat = self.prior_stem(ss_fea_masked)
        fused_feat = main_feat + prior_feat
        
        trunk_feat = self.trunk(fused_feat)
        internal_feat = self.internal_head(trunk_feat)
        fused_late_feat = self.fusion_module(internal_feat, ss_fea_masked)
        processed_feat = self.post_fusion_processor(fused_late_feat)
        logits = self.final_head(processed_feat)
        
        return logits   