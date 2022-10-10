from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
import torch.nn as nn
import torch

##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3, ngf=64, z_nc=512, img_f=512, L=6, layers=5, norm='none', activation='ReLU', use_spect=True,
             use_coord=False, init_type='orthogonal', gpu_ids=[]):

    net = ResEncoder(input_nc, ngf, z_nc, img_f, L, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)


def define_g(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='ReLU', output_scale=1,
             use_spect=True, use_coord=False, use_attn=True, init_type='orthogonal', gpu_ids=[]):

    net = ResGenerator(output_nc, ngf, z_nc, img_f, L, layers, norm, activation, output_scale, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


def define_d(input_nc=3, ndf=64, img_f=512, layers=6, norm='none', activation='LeakyReLU', use_spect=True, use_coord=False,
             use_attn=True,  model_type='ResDis', init_type='orthogonal', gpu_ids=[]):

    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)
    elif model_type == 'PatchDis':
        net = PatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)

def define_gmm(latent_size=256, k=4, activation='LeakyReLU', init_type='orthogonal', gpu_ids=[]):

    net = DynamicGMM(latent_size, k)

    return init_net(net, init_type, activation, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, L=6, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.downSampler = nn.AvgPool2d(2)
        self.down_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )

    def forward(self, img_m, img_c=None):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)
        
        f = feature[-1]
        extracted_batch = f.shape[0]
        f_to_z = self.downSampler(self.downSampler(f)).view(extracted_batch, -1)
        z = self.down_fc(f_to_z.squeeze())
        z_mu, z_logsigma = torch.chunk(z, 2, dim=1)
        z = z_mu + torch.randn_like(z_logsigma) * z_logsigma.exp()
        
        return z, z_mu, z_logsigma, feature


class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.up_convfc = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers - i - 1), img_f // ngf)
            if i > layers - output_scale:
                # upconv = ResBlock(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            else:
                # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'up', True)
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == 1 and use_attn:
                attn = Auto_Attn(ngf*mult, None)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        extracted_batch = z.shape[0]
        z = z.view(extracted_batch, 128, 2, 2)
        for i in range(2):
            z = self.up_convfc(F.interpolate(z, scale_factor=2))

        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i))
             f = generator(f)

        # the features come from mask regions and valid regions, we directly add them together
        out = f_m + f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i == 1 and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
                # out, attn = model(out)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            sequence +=[
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]

        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out

class DynamicGMM(nn.Module):
    def __init__(self, latent_size=256, k=4):
        super().__init__()
        latent_size = int(latent_size)
        self.latent_size = latent_size
        self.alpha = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.BatchNorm1d(latent_size // 2),
            nn.ReLU(),
            nn.Linear(latent_size // 2, k),
            nn.Softmax(dim=1)
        )
        self.mu = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.BatchNorm1d(latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * k),
        )
        self.sigma = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.BatchNorm1d(latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * k),
        )
        self.k = k
    
    def forward(self, zm, zc=None):
        if type(zc) != type(None):
            infer_ans = {}
            alpha = self.alpha(zm)
            z_c_mu_hat = self.mu(zm).view(-1, self.k, self.latent_size)
            z_c_logsigma_hat = self.sigma(zm).view(-1, self.k, self.latent_size)
            z_m_mu_hat = self.mu(zc).view(-1, self.k, self.latent_size)
            z_m_logsigma_hat = self.sigma(zc).view(-1, self.k, self.latent_size)
            infer_ans['alpha'] = alpha
            infer_ans['z_c_mu_hat'] = z_c_mu_hat
            infer_ans['z_c_logsigma_hat'] = z_c_logsigma_hat
            infer_ans['z_m_mu_hat'] = z_m_mu_hat
            infer_ans['z_m_logsigma_hat'] = z_m_logsigma_hat
            return infer_ans
        else:
            alpha = self.alpha(zm)
            mu = self.mu(zm).view(-1, self.k, self.latent_size)
            logsigma = self.sigma(zm).view(-1, self.k, self.latent_size)
            # sigma = torch.clamp(sigma, -2, 2)
            zc = mu + torch.randn_like(logsigma) * logsigma.exp()
            return zc, alpha