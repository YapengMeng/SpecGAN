import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.builder import build_component
from mmedit.models.common import InterpolatePack
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class SpecGAN(nn.Module):

    def __init__(self, in_size, out_size, img_in_channels=3, img_out_channels=3,
                 rdb_base_channels=128, num_rdb_blocks=23, style_channels=512,
                 num_mlps=8, channel_multiplier=2, blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01, default_style_mode='mix', eval_style_mode='single',
                 mix_prob=0.9, pretrained=None, bgr2rgb=False):

        super().__init__()

        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2)
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(True)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2 ** i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder_head = Fusion_Net()
        self.encoder = nn.ModuleList()
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [2 ** i for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))]
        # 32 64 128 256
        self.decoder = nn.ModuleList()
        for res in decoder_res:
            if res == in_size:
                in_channels = channels[res]
            else:
                in_channels = 2 * channels[res]

            if res < out_size:
                out_channels = channels[res * 2]
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        InterpolatePack(out_channels, out_channels, scale_factor=2, upsample_kernel=3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ))
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, 64, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(64, img_out_channels, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        # nn.Tanh(),
                    ))
        self.decoder_out = nn.Sequential(
                        nn.Conv2d(6, 3, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(3, img_out_channels, 3, 1, 1),
                        nn.Tanh(),
                    )


    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        h, w = lq.shape[2:]

        # encoder
        feat = lq
        encoder_features = []
        feat, decoder_feat = self.encoder_head(feat)
        encoder_features.append(feat)
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)  # B*num_styles*style_channels
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        generator_features = []
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            if out.size(2) <= self.in_size:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]

                out = torch.cat([out, feat], dim=1)
                out = self.fusion_out[fusion_index](out)

                skip = torch.cat([skip, feat], dim=1)
                skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            # store features for decoder
            if out.size(2) > self.in_size:
                generator_features.append(out)

            _index += 2

        hr = decoder_feat
        for i, block in enumerate(self.decoder):
            if i > 0:
                hr = torch.cat([hr, generator_features[i - 1]], dim=1)
            hr = block(hr)
        hr = self.decoder_out(torch.cat([hr, skip], dim=1))

        return hr

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super().__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB_Block(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super().__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G, kSize))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDB_branch(nn.Module):
    def __init__(self, num_RDB_block, growRate0, growRate, nConvLayers, kSize=3):
        super().__init__()

        G0 = growRate0
        G = growRate
        C = nConvLayers
        self.D = num_RDB_block

        # Redidual dense blocks and dense feature fusion RDB
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB_Block(growRate0=G0, growRate=G, nConvLayers=C)
            )
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, x, detail=False):
        x_in = x.clone()
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1)) + x_in
        if detail:
            return x, RDBs_out
        else:
            return x


class SimilarAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, mid_dim=64, **kwargs):
        super(SimilarAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, reference, x):
        """
        inputs :
            x : input feature maps(B X C X W X H)
        returns :
            out : self attention value
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, num_channel, width, height = reference.shape
        proj_query = self.query_conv(reference).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(reference).view(m_batchsize, -1, width * height)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, num_channel, width, height)
        return out


class Fusion_Net(nn.Module):

    def __init__(self, in_dim=(48, 3), out_dim=(512, 512), num_RDB_blocks=8, base_dim=128, growRate=64, Depth=8):
        super().__init__()

        self.out_dim = out_dim
        self.Relu = nn.LeakyReLU(negative_slope=0.2)
        self.Sig = nn.Sigmoid()

        self.layerInX_1 = nn.Conv2d(in_dim[0], base_dim, 3, 1, 1)
        self.layerInX_2 = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerInY_1 = nn.Conv2d(in_dim[1], base_dim, 3, 1, 1)
        self.layerInY_2 = nn.Conv2d(base_dim, base_dim, 3, 1, 1)

        # Shared Imformation extraction layer, between three input
        self.Infor = nn.Sequential(
            *[
                nn.Conv2d(base_dim * 2 + 2, base_dim, 3, 1, 1),
                self.Relu,
                nn.Conv2d(base_dim, base_dim, 3, 1, 1),
                self.Relu,
                nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            ]
        )

        self.layerX_S = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerX_M = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerX_B = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerY_S = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerY_M = nn.Conv2d(base_dim, base_dim, 3, 1, 1)
        self.layerY_B = nn.Conv2d(base_dim, base_dim, 3, 1, 1)

        self.similarX = SimilarAttn(base_dim)
        self.similarY = SimilarAttn(base_dim)

        self.layerOut_X = nn.Sequential(
            nn.Conv2d(base_dim * 2, self.out_dim[0], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_dim[0], self.out_dim[0], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_dim[0], self.out_dim[0], 3, 1, 1),
        )
        self.layerOut_Y = nn.Sequential(
            nn.Conv2d(base_dim * 2, self.out_dim[1], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_dim[1], self.out_dim[1], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.out_dim[1], self.out_dim[1], 3, 1, 1),
        )

        # backbone of three branch in two stage
        self.branch1_1 = RDB_branch(num_RDB_block=num_RDB_blocks, growRate0=base_dim, growRate=growRate,
                                    nConvLayers=Depth)
        self.branch1_2 = RDB_branch(num_RDB_block=num_RDB_blocks, growRate0=base_dim, growRate=growRate,
                                    nConvLayers=Depth)
        self.branch2_1 = RDB_branch(num_RDB_block=num_RDB_blocks, growRate0=base_dim, growRate=growRate,
                                    nConvLayers=Depth)
        self.branch2_2 = RDB_branch(num_RDB_block=num_RDB_blocks, growRate0=base_dim, growRate=growRate,
                                    nConvLayers=Depth)

    def forward(self, xy):

        hsi = xy[:, :-3, :, :]
        rgb = xy[:, -3:, :, :]

        # Make mask layers to distinguish RGB layers and HSI layers
        a = torch.ones(rgb.shape[2], rgb.shape[3]).unsqueeze(0)
        b = torch.zeros(rgb.shape[2], rgb.shape[3]).unsqueeze(0)
        self.MaskX = torch.cat((b, a), 0).unsqueeze(0).cuda()
        self.MaskY = torch.cat((a, b), 0).unsqueeze(0).cuda()

        # Input shallow processing
        out_In1 = self.layerInX_1(hsi)
        out_In2 = self.layerInY_1(rgb)
        out_1 = self.layerInX_2(out_In1)
        out_2 = self.layerInY_2(out_In2)

        # Deep processing 1
        out_1 = self.branch1_1(out_1)
        out_2 = self.branch2_1(out_2)

        # Fusion processing 2
        Infor_X = self.Infor(torch.cat((out_1, out_2, self.MaskX.repeat(out_1.shape[0], 1, 1, 1)), 1))
        out1_S = self.similarX(self.layerX_S(Infor_X), out_1)
        out1_M = self.layerX_M(Infor_X)
        out1_B = self.layerX_B(Infor_X)
        out_1 = out1_S + out_1 * out1_M + out1_B
        Infor_Y = self.Infor(torch.cat((out_1, out_2, self.MaskY.repeat(out_1.shape[0], 1, 1, 1)), 1))
        out2_S = self.similarY(self.layerY_S(Infor_Y), out_2)
        out2_M = self.layerY_M(Infor_Y)
        out2_B = self.layerY_B(Infor_Y)
        out_2 = out2_S + out_2 * out2_M + out2_B

        # Deep processing 2
        out_1 = self.branch1_2(out_1)
        out_2 = self.branch2_2(out_2)

        # Fusion processing 3
        Infor_X = self.Infor(torch.cat((out_1, out_2, self.MaskX.repeat(out_1.shape[0], 1, 1, 1)), 1))
        out1_S = self.similarX(self.layerX_S(Infor_X), out_1)
        out1_M = self.layerX_M(Infor_X)
        out1_B = self.layerX_B(Infor_X)
        out_1 = out1_S + out_1 * out1_M + out1_B

        Infor_Y = self.Infor(torch.cat((out_1, out_2, self.MaskY.repeat(out_1.shape[0], 1, 1, 1)), 1))
        out2_S = self.similarY(self.layerY_S(Infor_Y), out_2)
        out2_M = self.layerY_M(Infor_Y)
        out2_B = self.layerY_B(Infor_Y)
        out_2 = out2_S + out_2 * out2_M + out2_B

        # Output processing
        out_X = self.layerOut_X(torch.cat((out_1 + out_In1, out_2 + out_In2), 1))
        out_Y = self.layerOut_Y(torch.cat((out_1 + out_In1, out_2 + out_In2), 1))

        return out_X, out_Y
