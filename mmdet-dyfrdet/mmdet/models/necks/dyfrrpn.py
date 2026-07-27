import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList, auto_fp16
from mmcv.runner.fp16_utils import force_fp32
from ..builder import NECKS, build_backbone
from .fpn import FPN
from .rfp import ASPP, RFP
from mmcv.cnn import ConvModule

@NECKS.register_module()
class DyFrFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 win_size = 38,
                 beta = 0.5,
                 relative = False,
                 start_epoch = 8):
        super(DyFrFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.relative = relative
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.win_size = win_size
        self.beta = beta
        self.start_epoch = start_epoch

        self.sfconv = nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size = 1)
        self.fsfconv = nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1)
        self.cross_attn = CrossAttention(self.out_channels, num_heads = 8)
        self.alpha_l_proj = nn.Linear(self.out_channels * self.win_size * self.win_size, self.out_channels)
        self.band_del_width_proj = nn.Linear(self.out_channels * self.win_size * self.win_size, self.out_channels)
        self.relu6 = nn.ReLU6()
        self.phase_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
        )
        self.amp_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
        )

        self.init_small_kaiming_(self)

    def init_small_kaiming_(self, model, scale=0.01):
        """
        适用于 Softplus 的小权重初始化，不设置 bias，只缩放权重。
        """
        # alpha_l_proj
        nn.init.kaiming_uniform_(model.alpha_l_proj.weight, a=0)  # ReLU 风格也适合 Softplus
        model.alpha_l_proj.weight.data.mul_(scale)

        # band_width_proj
        nn.init.kaiming_uniform_(model.band_del_width_proj.weight, a=0)
        model.band_del_width_proj.weight.data.mul_(scale)

    @auto_fp16()
    def forward(self, inputs, epoch = 48):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        if epoch < self.start_epoch:
            return outs
        filtered_outs = []

        #energy_keep_ratios = []

        for lv in range(len(outs)):
            feature_fp32 = outs[lv].to(torch.float32)
            ffts = torch.fft.fft2(feature_fp32, norm='ortho')
            if self.relative:
                alpha_l, alpha_h = self.predict_alpha_relative(feature_fp32, ffts)
            else:
                alpha_l, alpha_h = self.predict_alpha(feature_fp32, ffts)
            masked_ffts = self.fft2_filter_with_box_mask(ffts, alpha_l, alpha_h)
            newout = torch.fft.ifft2(masked_ffts, norm='ortho').real
            filtered_outs.append(outs[lv] - self.beta * newout.half())

        return tuple(filtered_outs)

    def fft_feature_stack(self, fft_feat):
        ff_real = fft_feat.real  # B x C x win x win
        ff_imag = fft_feat.imag  # B x C x win x win
        ff = torch.cat([ff_real, ff_imag], dim=1)
        return ff  

    def get_amp_phase(self, fft_feat):
        amplitude = torch.abs(fft_feat)  # B x C x H x W
        phase = torch.angle(fft_feat)# B x C x H x W
        return amplitude, phase
    
    def avg_max_pool_combine(self, x, target_size=38, method="concat"):
        """
        将 B, C, H, W 的特征同时进行 avg/max pooling 到 B, C, target_size, target_size
        method: 'concat' 或 'sum'
        """
        avg_pool = F.adaptive_avg_pool2d(x, (target_size, target_size))
        max_pool = F.adaptive_max_pool2d(x, (target_size, target_size))

        if method == "concat":
            return torch.cat([avg_pool, max_pool], dim=1)  # (B, 2C, target_size, target_size)
        elif method == "sum":
            return avg_pool + max_pool  # (B, C, target_size, target_size)
        else:
            raise ValueError("method 应为 'concat' 或 'sum'")
    
    @force_fp32(apply_to=('freq_feature',))
    def predict_alpha(self, space_feature, freq_feature):
        #produce space_feature
        sf = self.sfconv(self.avg_max_pool_combine(space_feature, target_size = self.win_size, method = "concat"))
        #produce freq_feature
        amp, phase = self.get_amp_phase(freq_feature)
        amp = self.amp_conv(amp)
        phase = self.phase_conv(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        complex_spectrum = torch.complex(real, imag)
        fsf = torch.fft.ifft2(complex_spectrum, norm='ortho').real
        fsf = self.fsfconv(self.avg_max_pool_combine(fsf, target_size=self.win_size, method="concat"))
        fsf = fsf + self.cross_attn(fsf, sf)
        fsf = fsf.flatten(1)
        alpha_l = self.relu6(self.alpha_l_proj(fsf)) / 6.0
        sz = 1 - self.relu6(self.band_del_width_proj(fsf)) / 6.0
        alpha_h = alpha_l + sz
        alpha_h = torch.clamp(alpha_h, 0.0, 1.0) 
        alpha_l = torch.clamp(alpha_l, 0.0, 1.0) 
        return alpha_l, alpha_h
    
    @force_fp32(apply_to=('freq_feature',))
    def predict_alpha_relative(self, space_feature, freq_feature):
        # produce space_feature
        sf = self.sfconv(self.avg_max_pool_combine(space_feature, target_size=self.win_size, method="concat"))
        # produce freq_feature
        amp, phase = self.get_amp_phase(freq_feature)
        amp = self.amp_conv(amp)
        phase = self.phase_conv(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        complex_spectrum = torch.complex(real, imag)
        fsf = torch.fft.ifft2(complex_spectrum, norm='ortho').real
        fsf = self.fsfconv(self.avg_max_pool_combine(fsf, target_size=self.win_size, method="concat"))
        fsf = fsf + self.cross_attn(fsf, sf)
        fsf = fsf.flatten(1)
        alpha_l = 0.05 * torch.exp(self.alpha_l_proj(fsf))
        alpha_h = 0.95 * torch.exp(self.band_del_width_proj(fsf))
        alpha_h = torch.clamp(alpha_h, 0.0, 1.0) 
        alpha_l = torch.clamp(alpha_l, 0.0, 1.0)  
        return alpha_l, alpha_h
    
    def fft2_filter_with_box_mask(self, ffts, alpha_l, alpha_h):
        B, C, H, W = ffts.shape
        assert H == W, "H not equal W"
        device = ffts.device

        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

        lower = (alpha_l * H).view(B, C, 1, 1)  # (B, C, 1, 1)
        upper = (alpha_h * H).view(B, C, 1, 1)  # (B, C, 1, 1)

        low_low = (xx < lower) & (yy < lower)  
        high_high = (xx > upper) & (yy > upper) 
        mask = (low_low | high_high)  
        mask = mask.float()  

        ffts_filtered = ffts * mask

        return ffts_filtered


@NECKS.register_module()
class DyFrFPN_v2(FPN):
    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 aspp_dilations=(1, 3, 6, 1),
                 init_cfg=None,
                 win_size = 13,
                 beta = 0.5,
                 relative = False,
                 start_epoch = 24,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.win_size = win_size
        self.beta = beta
        self.start_epoch = start_epoch

        self.sfconv = nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1)
        self.fsfconv = nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1)
        self.cross_attn = CrossAttention(self.out_channels, num_heads=8)
        self.alpha_l_proj = nn.Linear(self.out_channels * self.win_size * self.win_size, self.out_channels)
        self.band_del_width_proj = nn.Linear(self.out_channels * self.win_size * self.win_size, self.out_channels)
        self.relu6 = nn.ReLU6()
        self.phase_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
        )
        self.amp_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
        )
        self.relative = relative
        self.init_small_kaiming_(self)

    def init_small_kaiming_(self, model, scale=0.01):
        """
        适用于 Softplus 的小权重初始化，不设置 bias，只缩放权重。
        """
        # alpha_l_proj
        nn.init.kaiming_uniform_(model.alpha_l_proj.weight, a=0)  # ReLU 风格也适合 Softplus
        model.alpha_l_proj.weight.data.mul_(scale)

        # band_width_proj
        nn.init.kaiming_uniform_(model.band_del_width_proj.weight, a=0)
        model.band_del_width_proj.weight.data.mul_(scale)

    def init_weights(self):
        # Avoid using super().init_weights(), which may alter the default
        # initialization of the modules in self.rfp_modules that have missing
        # keys in the pretrained checkpoint.
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs, epoch):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) + 1  # +1 for input image
        img = inputs.pop(0)
        # FPN forward
        x = super().forward(tuple(inputs))
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [x[0]] + list(
                self.rfp_aspp(x[i]) for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            # FPN forward
            x_idx = super().forward(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx])
            x = x_new


        if epoch < self.start_epoch:
            return x
        filtered_outs = []

        #energy_keep_ratios = []

        for lv in range(len(x)):
            feature_fp32 = x[lv].to(torch.float32)
            ffts = torch.fft.fft2(feature_fp32, norm='ortho')
            if self.relative:
                alpha_l, alpha_h = self.predict_alpha_relative(feature_fp32, ffts)
            else:
                alpha_l, alpha_h = self.predict_alpha(feature_fp32, ffts)
            masked_ffts = self.fft2_filter_with_box_mask(ffts, alpha_l, alpha_h)
            newout = torch.fft.ifft2(masked_ffts, norm='ortho').real
            filtered_outs.append(x[lv] - self.beta * newout.half())

        return filtered_outs

    def fft_feature_stack(self, fft_feat):
        ff_real = fft_feat.real  # B x C x win x win
        ff_imag = fft_feat.imag  # B x C x win x win
        ff = torch.cat([ff_real, ff_imag], dim=1)
        return ff 

    def get_amp_phase(self, fft_feat):
        amplitude = torch.abs(fft_feat)  # B x C x H x W
        phase = torch.angle(fft_feat)# B x C x H x W
        return amplitude, phase
    
    def avg_max_pool_combine(self, x, target_size=38, method="concat"):
        avg_pool = F.adaptive_avg_pool2d(x, (target_size, target_size))
        max_pool = F.adaptive_max_pool2d(x, (target_size, target_size))

        if method == "concat":
            return torch.cat([avg_pool, max_pool], dim=1)  # (B, 2C, target_size, target_size)
        elif method == "sum":
            return avg_pool + max_pool  # (B, C, target_size, target_size)
        else:
            raise ValueError("method 应为 'concat' 或 'sum'")
    
    @force_fp32(apply_to=('freq_feature',))
    def predict_alpha(self, space_feature, freq_feature):
        #produce space_feature
        sf = self.sfconv(self.avg_max_pool_combine(space_feature, target_size = self.win_size, method = "concat"))
        #produce freq_feature
        amp, phase = self.get_amp_phase(freq_feature)
        amp = self.amp_conv(amp)
        phase = self.phase_conv(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        complex_spectrum = torch.complex(real, imag)
        fsf = torch.fft.ifft2(complex_spectrum, norm='ortho').real
        fsf = self.fsfconv(self.avg_max_pool_combine(fsf, target_size=self.win_size, method="concat"))
        fsf = fsf + self.cross_attn(fsf, sf)
        fsf = fsf.flatten(1)
        alpha_l = self.relu6(self.alpha_l_proj(fsf)) / 6.0
        sz = 1 - self.relu6(self.band_del_width_proj(fsf)) / 6.0
        alpha_h = alpha_l + sz
        alpha_h = torch.clamp(alpha_h, 0.0, 1.0) 
        alpha_l = torch.clamp(alpha_l, 0.0, 1.0) 
        return alpha_l, alpha_h
   
    @force_fp32(apply_to=('freq_feature',))
    def predict_alpha_relative(self, space_feature, freq_feature):
        # produce space_feature
        sf = self.sfconv(self.avg_max_pool_combine(space_feature, target_size=self.win_size, method="concat"))
        # produce freq_feature
        amp, phase = self.get_amp_phase(freq_feature)
        amp = self.amp_conv(amp)
        phase = self.phase_conv(phase)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        complex_spectrum = torch.complex(real, imag)
        fsf = torch.fft.ifft2(complex_spectrum, norm='ortho').real
        fsf = self.fsfconv(self.avg_max_pool_combine(fsf, target_size=self.win_size, method="concat"))
        fsf = fsf + self.cross_attn(fsf, sf)
        fsf = fsf.flatten(1)
        alpha_l = 0.05 * torch.exp(self.alpha_l_proj(fsf))
        alpha_h = 0.95 * torch.exp(self.band_del_width_proj(fsf))
        alpha_h = torch.clamp(alpha_h, 0.0, 1.0)  
        alpha_l = torch.clamp(alpha_l, 0.0, 1.0)  
        return alpha_l, alpha_h
    
    def fft2_filter_with_box_mask(self, ffts, alpha_l, alpha_h):
        B, C, H, W = ffts.shape
        assert H == W, "H not equal W"
        device = ffts.device

        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))

        lower = (alpha_l * H).view(B, C, 1, 1)  # (B, C, 1, 1)
        upper = (alpha_h * H).view(B, C, 1, 1)  # (B, C, 1, 1)

        low_low = (xx < lower) & (yy < lower) 
        high_high = (xx > upper) & (yy > upper)  
        mask = (low_low | high_high)  
        mask = mask.float() 

        ffts_filtered = ffts * mask

        return ffts_filtered

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, ff, sf):
        B, C, Hf, Wf = ff.shape  # ff: B x C x win x win
        _, _, Hs, Ws = sf.shape  # sf: B x C x 38 x 38

        # Flatten spatial dims
        ff_flat = ff.flatten(2).transpose(1, 2)  # B x (win*win) x C
        sf_flat = sf.flatten(2).transpose(1, 2)  # B x (38*38) x C

        # Linear projections
        Q = self.q_proj(ff_flat)  # B x Nq x C
        K = self.k_proj(sf_flat)  # B x Nk x C
        V = self.v_proj(sf_flat)  # B x Nk x C

        # Split heads
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B x heads x Nq x d
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B x heads x Nk x d
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B x heads x Nk x d

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B x heads x Nq x Nk
        attn_weights = F.softmax(attn_scores, dim=-1)  # B x heads x Nq x Nk
        attn_output = torch.matmul(attn_weights, V)  # B x heads x Nq x d

        # Combine heads
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, self.dim)  # B x Nq x C

        # Final projection
        output = self.out_proj(attn_output)  # B x Nq x C

        # Reshape back to spatial
        output = output.transpose(1, 2).view(B, C, Hf, Wf)  # B x C x win x win

        return output