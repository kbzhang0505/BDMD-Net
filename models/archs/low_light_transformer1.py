import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


from models.archs.transformer.Models import Encoder_patch66

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x) #104
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)#256
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, y, z, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # 8,64,64,32
        y = y.permute(0, 2, 3, 1)  # 8,64,64,32
        z = z.permute(0, 2, 3, 1)  # 8,64,64,32
        for (attn, ff) in self.blocks:
            x = attn(x, y, z, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, y_in, z_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        z = z_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(y)
        v_inp = self.to_v(z)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

###############################
class low_light_transformer(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA


        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, 2 * nf, 3, 2, 1, bias=True)
        self.conv_sec_2 = nn.Conv2d(3, 2 * nf, 3, 1, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(2 * nf, 4 * nf, 3, 2, 1, bias=True)
        self.conv_sec_3 = nn.Conv2d(3, 4 * nf, 3, 1, 1, bias=True)

        # self.feature_extraction1 = Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction2 = Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction3 = Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction4 = Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction5 = Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction6 = Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction11 = Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction22 = Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction33 = Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction44 = Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction55 = Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4)
        # self.feature_extraction66 = Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4)

        # self.se_layer1 = SELayer(64, 8)
        # self.se_layer2 = SELayer(128, 8)
        # self.se_layer3 = SELayer(256, 8)
        self.processmag1 = nn.Sequential(
            SELayer(64, 8),
            Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4),
            SELayer(64, 8),
            Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4))
        self.processmag2 = nn.Sequential(
            SELayer(128, 8),
            Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4),
            SELayer(128, 8),
            Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4))
        self.processmag3 = nn.Sequential(
            SELayer(256, 8),
            Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4),
            SELayer(256, 8),
            Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4))
        self.processpha1 = nn.Sequential(
            SELayer(64, 8),
            Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4),
            SELayer(64, 8),
            Bottle2neck(64, 64, downsample=None, baseWidth=26, scale=4))
        self.processpha2 = nn.Sequential(
            SELayer(128, 8),
            Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4),
            SELayer(128, 8),
            Bottle2neck(128, 128, downsample=None, baseWidth=26, scale=4))
        self.processpha3 = nn.Sequential(
            SELayer(256, 8),
            Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4),
            SELayer(256, 8),
            Bottle2neck(256, 256, downsample=None, baseWidth=26, scale=4))

        self.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, bias=True)
        # self.conv2 = nn.Conv2d(129, 128, kernel_size=1, stride=1, bias=True)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)



        self.depth_conv1 = nn.Conv2d(
            64, 64, kernel_size=5, padding=2, bias=True, groups=4)
        self.depth_conv2 = nn.Conv2d(
            128, 128, kernel_size=5, padding=2, bias=True, groups=4)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fpre1 = nn.Conv2d(nf, nf, 1, 1, 0)
        self.fpre2 = nn.Conv2d(nf*2, nf*2, 1, 1, 0)
        self.fpre3 = nn.Conv2d(nf*4, nf*4, 1, 1, 0)

        self.down_1 = Downsample(int(nf))
        self.down_2 = Downsample(int(nf * 2))

        self.up1 = Upsample(int(nf * 4))
        self.up2 = Upsample(int(nf * 2))

        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.transformer1 = IGAB(dim=64, dim_head=64, heads=1, num_blocks=4)
        self.transformer2 = IGAB(dim=128, dim_head=128, heads=1, num_blocks=4)
        self.transformer3 = IGAB(dim=256, dim_head=256, heads=1, num_blocks=4)

        self.fusion_layer = AdaptiveWeightFusionFreqDomain(nf)


    def forward(self, x, x2, x3, mask=None, mask2=None, mask4=None):
        x_center = x
        B, C, H, W = x.shape
        L1_fea_1 = self.conv_first_1(x_center)  # 64，128
        L1_fea_x2 = self.conv_sec_2(x2)
        L1_fea_2 = self.conv_first_2(L1_fea_1) + L1_fea_x2  # 128，64
        L1_fea_x3 = self.conv_sec_3(x3)
        L1_fea_3 = self.conv_first_3(L1_fea_2) + L1_fea_x3  # 256，32

        mean_c1 = x_center.mean(dim=1).unsqueeze(1)  # 1,128
        input1 = torch.cat([x_center, mean_c1], dim=1)  # 4,128
        x_1 = self.conv1(input1)  # 64,128
        illu_fea1 = self.depth_conv1(x_1)  # 64,128
        L1_T_1 = self.transformer1(L1_fea_1, L1_fea_1, L1_fea_1, illu_fea1)
        L1_T_1 = self.transformer1(L1_T_1, L1_T_1, L1_T_1, illu_fea1)
        L1_T_1 = self.transformer1(L1_T_1, L1_T_1, L1_T_1, illu_fea1)
        illu_fea2 = self.down_1(illu_fea1)

        L1_T_2 = self.transformer2(L1_fea_x2, L1_fea_x2, L1_fea_x2, illu_fea2)
        L1_T_2 = self.transformer2(L1_T_2, L1_fea_2, L1_fea_2, illu_fea2)
        q1 = self.down_1(L1_T_1) + L1_T_2
        L1_T_2 = self.transformer2(q1, L1_T_2, L1_T_2, illu_fea2)

        illu_fea3 = self.down_2(illu_fea2)
        L1_T_3 = self.transformer3(L1_fea_x3, L1_fea_x3, L1_fea_x3, illu_fea3)
        L1_T_3 = self.transformer3(L1_T_3, L1_fea_3, L1_fea_3, illu_fea3)
        q2 = self.down_2(L1_T_2) + L1_T_3
        L1_T_3 = self.transformer3(q2, L1_T_3, L1_T_3, illu_fea3)
        L1_T_3 = self.transformer3(L1_T_3, L1_T_3, L1_T_3, illu_fea3)
        q3 = self.up1(L1_T_3) + L1_T_2
        L1_T_2 = self.transformer2(q3, L1_T_2, L1_T_2, illu_fea2)
        q4 = self.up2(L1_T_2) + L1_T_1
        L1_T_1 = self.transformer1(q4, L1_T_1, L1_T_1, illu_fea1)

        # fea_T = L1_T_1 + L1_fea_1 + self.up2(L1_T_2 + L1_fea_x2)
        L1_T_2 = self.up2(L1_T_2)
        L1_T_3 = self.up2(self.up1(L1_T_3))
        # fea_T = self.fusion_layer(L1_T_1,L1_T_2,L1_T_3)
        fea_T = L1_T_1 + L1_T_2 + L1_T_3
        # fea_T = self.lrelu(self.conv3(fea_T))

        ttf1 = torch.fft.fft2(self.fpre1(L1_fea_1), norm='backward')
        mag1, pha1 = torch.abs(ttf1), torch.angle(ttf1)
        mag1 = self.processmag1(mag1)  # 64，128
        pha1 = self.processpha1(pha1)
        # freq1 = mag1 * torch.exp(1j * pha1)
        # freq1 = torch.fft.irfft2(freq1, s=(H, W))

        ttf2 = torch.fft.fft2(self.fpre2(L1_fea_2), norm='backward')
        mag2, pha2 = torch.abs(ttf2), torch.angle(ttf2)
        mag2 = self.processmag2(mag2)
        pha2 = self.processpha2(pha2)  # 128，64
        # freq2 = mag2 * torch.exp(1j * pha2)
        # freq2 = torch.fft.irfft2(freq2, s=(H//2, W//2))

        ttf3 = torch.fft.fft2(self.fpre3(L1_fea_3), norm='backward')
        mag3, pha3 = torch.abs(ttf3), torch.angle(ttf3)
        mag3 = self.processmag3(mag3)
        pha3 = self.processpha3(pha3)  # 256，32
        # freq3 = mag3 * torch.exp(1j * pha3)
        # freq3 = torch.fft.irfft2(freq3, s=(H//4, W//4))

        mag2 = self.up2(mag2)
        pha2 = self.up2(pha2)
        mag3 = self.up2(self.up1(mag3))
        pha3 = self.up2(self.up1(pha3))
        fea = self.fusion_layer(L1_fea_1, mag1, pha1, mag2, pha2, mag3, pha3)
        channel = L1_fea_1.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_T * (1 - mask) + fea * mask


        # fea = self.lrelu(self.conv4(self.up2(fea2 + self.up1(fea3)) + fea1))
        # fea2 = self.up2(fea2)
        # fea3 = self.up2(self.up1(fea3))
        # # fea = fea1 + fea2 + fea3
        # fea = self.fusion_layer(fea1,fea2,fea3)

        # channel = fea.shape[1]
        # mask = mask.repeat(1, channel, 1, 1)
        # fea = fea_T * (1 - mask) + fea * mask


        out_noise = self.conv_last(fea) + x_center

        return out_noise


class AdaptiveWeightFusionFreqDomain(nn.Module):
    def __init__(self, in_dim):
        super(AdaptiveWeightFusionFreqDomain, self).__init__()
        self.reweight_magnitude = nn.Sequential(
            nn.Linear(3 * in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 3 * in_dim)
        )
        self.reweight_phase = nn.Sequential(
            nn.Linear(3 * in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 3 * in_dim)
        )

    def forward(self, x1, mag1, pha1, mag2, pha2, mag3, pha3):
        B, C, H, W = x1.shape

        # x1_freq = torch.fft.rfft2(x1)
        # x2_freq = torch.fft.rfft2(x2)
        # x3_freq = torch.fft.rfft2(x3)
        #
        # # Extract magnitude and phase
        # mag1, pha1 = torch.abs(x1_freq), torch.angle(x1_freq)
        # mag2, pha2 = torch.abs(x2_freq), torch.angle(x2_freq)
        # mag3, pha3 = torch.abs(x3_freq), torch.angle(x3_freq)

        # Adaptive average pooling and reweighting
        mag_u = F.adaptive_avg_pool2d(mag1, output_size=1).view(B, C)
        mag_v = F.adaptive_avg_pool2d(mag2, output_size=1).view(B, C)
        mag_w = F.adaptive_avg_pool2d(mag3, output_size=1).view(B, C)

        pha_u = F.adaptive_avg_pool2d(pha1, output_size=1).view(B, C)
        pha_v = F.adaptive_avg_pool2d(pha2, output_size=1).view(B, C)
        pha_w = F.adaptive_avg_pool2d(pha3, output_size=1).view(B, C)

        concat_mag = torch.cat([mag_u, mag_v, mag_w], dim=1)
        weights_mag = self.reweight_magnitude(concat_mag).view(B, 3, C).permute(1, 0, 2).softmax(dim=0)
        weights_mag = weights_mag.unsqueeze(-1).unsqueeze(-1)

        concat_pha = torch.cat([pha_u, pha_v, pha_w], dim=1)
        weights_pha = self.reweight_phase(concat_pha).view(B, 3, C).permute(1, 0, 2).softmax(dim=0)
        weights_pha = weights_pha.unsqueeze(-1).unsqueeze(-1)

        # Apply weights
        fused_mag = mag1 * weights_mag[0] + mag2 * weights_mag[1] + mag3 * weights_mag[2]
        fused_pha = pha1 * weights_pha[0] + pha2 * weights_pha[1] + pha3 * weights_pha[2]

        # Combine magnitude and phase back into complex representation
        fused_freq = fused_mag * torch.exp(1j * fused_pha)

        # Convert back to spatial domain
        fused_spatial = torch.fft.ifft2(fused_freq, s=(H, W))
        fused_spatial = torch.real(fused_spatial)
        return fused_spatial