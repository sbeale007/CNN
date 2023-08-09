# Borrowed from:
# https://github.com/Lornatang/SRGAN-PyTorch/
# Adopted to ESRGAN: https://arxiv.org/abs/1809.00219

import torch
import torch.nn as nn

class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr

class Conv2dDepthwiseSeparable(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super(Conv2dDepthwiseSeparable, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Generator(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(
        self,
        filters,
        fine_dims,
        channels,
        n_predictands=1,
        num_res_blocks=16,
        num_upsample=3,
    ):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, n_predictands, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        # out[:, 0, ...] = nn.ReLU()(
        #     out[:, 0, ...]
        # )  # I tried this just to clip to positive values, but it didn't seem to work. might be a more clever way
        return out
    
class Generator_hr_cov(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(
        self,
        filters,
        fine_dims,
        channels,
        channels_hr_cov=1,
        n_predictands=1,
        num_res_blocks=16,
        num_res_blocks_fine=1,
        num_upsample=3,
    ):
        super(Generator_hr_cov, self).__init__()
        self.fine_res = fine_dims
        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(
            channels_hr_cov, filters, kernel_size=3, stride=1, padding=1
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )
        self.res_blocksf = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks_fine)]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.LR_pre = nn.Sequential(
            self.conv1, ShortcutBlock(nn.Sequential(self.res_blocks, self.conv2))
        )
        self.HR_pre = nn.Sequential(
            self.conv1f, ShortcutBlock(nn.Sequential(self.res_blocksf, self.conv2))
        )

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            # nn.Conv2d(filters * 2, filters + 1, kernel_size=3, stride=1, padding=1),
            # ResidualInResidualDenseBlock(filters + 1),
            nn.Conv2d(filters * 2, filters + 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters + 1, n_predictands, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_coarse, x_fine):
        out = self.LR_pre(x_coarse)  ## LR branch
        outc = self.upsampling(out)
        outf = self.HR_pre(x_fine)  ## HR branch
        out = torch.cat((outc, outf), 1)  ##combine
        out = self.conv3(out)
        return out
    
class Generator_lr_global(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(
        self,
        filters,
        fine_dims,
        channels,
        channels_hr_cov=1,
        n_predictands=1,
        num_res_blocks=16,
        num_res_blocks_fine=1,
        num_upsample=3,
    ):
        super(Generator_lr_global, self).__init__()
        self.fine_res = fine_dims
        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(
            channels_hr_cov, filters, kernel_size=3, stride=1, padding=1
        )
        self.conv1l = nn.Conv2d(channels, filters, kernel_size=3, stride =1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )
        self.res_blocksf = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks_fine)]
        )
        self.res_blocksl = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        )

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.LR_pre = nn.Sequential(
            self.conv1, ShortcutBlock(nn.Sequential(self.res_blocks, self.conv2))
        )
        self.HR_pre = nn.Sequential(
            self.conv1f, ShortcutBlock(nn.Sequential(self.res_blocksf, self.conv2))
        )
        self.LR_large = nn.Sequential(
            self.conv1l, ShortcutBlock(nn.Sequential(self.res_blocksl, self.conv2))
        )

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            # nn.Conv2d(filters * 2, filters + 1, kernel_size=3, stride=1, padding=1),
            # ResidualInResidualDenseBlock(filters + 1),
            nn.Conv2d(filters * 2, filters + 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters + 1, n_predictands, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_small, x_large, x_fine):
        out = self.LR_pre(x_small)  ## LR branch
        outc = self.upsampling(out)
        out = self.LR_large(x_large)
        outl = self.upsampling(out)
        outf = self.HR_pre(x_fine)  ## HR branch
        out = torch.cat((outc, outl, outf), 1)  ##combine
        out = self.conv3(out)
        return out


class Critic(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self, lr_dim, hr_dim, n_predictands):
        super(Critic, self).__init__()
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.n_predictands = n_predictands

        self.features = nn.Sequential(
            nn.Conv2d(
                self.n_predictands, self.lr_dim, kernel_size=3, stride=1, padding=1
            ),  # input is (3) x 96 x 96
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                self.lr_dim,
                self.lr_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. (64) x 48 x 48
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                self.lr_dim,
                2 * self.lr_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.lr_dim,
                2 * self.lr_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. (128) x 24 x 24
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                2 * self.lr_dim,
                4 * self.lr_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                4 * self.lr_dim,
                4 * self.lr_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. (256) x 12 x 12
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                4 * self.lr_dim,
                8 * self.lr_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                8 * self.lr_dim,
                8 * self.lr_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),  # state size. (512) x 6 x 6
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(8*self.lr_dim, 16*self.lr_dim, kernel_size=3, stride=1, padding=1, bias=False),  # state size. (512) x 6 x 6
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(16*self.lr_dim, 16*self.lr_dim, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (1024) x 3 x 3
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int((self.lr_dim * 2**3) * (self.hr_dim / 2**4) ** 2), 100),
            # nn.Linear(32*24 * self.lr_dim, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.features(input)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
