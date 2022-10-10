import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task
import itertools
import torch.nn.functional as F

class Pluralistic(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "Pluralistic Image Completion with Gaussion Mixture Model"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            # parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['rec', 'ad_g', 'ad_d', 'freq', 'bm']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out', 'img_g']
        self.model_names = ['E', 'G', 'D', 'GMM']
        self.k = opt.k
        self.L2loss = torch.nn.MSELoss()
        self.L1loss = torch.nn.L1Loss()

        # define the inpainting model
        self.net_E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=opt.output_scale,
                                      norm='instance', activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the dynamic GMM model
        self.net_GMM = network.define_gmm(latent_size=256, k=opt.k, activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                        filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_GMM = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net_GMM.parameters()),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_GMM)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_c = (1 - self.mask) * self.img_truth

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)
    
    def test(self):
        '''output GMM diverse results'''
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        # encoder process
        z, z_mu, z_logsigma, f = self.net_E(self.img_m, self.img_c)

        z_m, z_c, f_m, f_e, mask = self.get_G_inputs(z, f)

        for i in range(self.opt.sample_num):
            z_c_gmm, alpha = self.net_GMM(z_m.detach())
            index = torch.distributions.Categorical(alpha).sample()
            index = index.unsqueeze(1).unsqueeze(1).repeat(1, 1, 256)
            z_c_latent = torch.gather(z_c_gmm, 1, index).squeeze(1)
            z = torch.cat([z_m, z_c_latent], dim=1)
            results, attn = self.net_G(z, f_m, f_e, mask)
            img_out = (1 - self.mask) * results[-1].detach() + self.mask * self.img_truth
            self.save_results(img_out, i, data_name='out')
    
    def testGMM(self, kk):
        '''test GMM'''
        # encoder process
        z, z_mu, z_logsigma, f = self.net_E(self.img_m, self.img_c)

        z_m, z_c, f_m, f_e, mask = self.get_G_inputs(z, f)
        # gmm
        z_c_gmm, _ = self.net_GMM(z_m.detach())
        z_c_latent = z_c_gmm[:, kk, :].squeeze()

        z = torch.cat([z_m, z_c_latent], dim=1)
        results, attn = self.net_G(z, f_m, f_e, mask)

        img_out = (1-self.mask) * results[-1].detach() + self.mask * self.img_truth

        return img_out
    
    def get_G_inputs(self, z, f):
        """Process the encoder feature and distributions for generation network"""
        z_m = z.chunk(2)[0]
        z_c = z.chunk(2)[1]
        f_m = f[-1].chunk(2)[0]
        f_e = f[2].chunk(2)[0]
        scale_mask = task.scale_img(self.mask, size=[f_e.size(2), f_e.size(3)])
        mask = scale_mask.chunk(3,dim=1)[0]
        return z_m, z_c, f_m, f_e, mask
    
    def get_z_c_hat(self, z_m, z_c):
        infer_ans = self.net_GMM(z_m.detach(), z_c.detach())
        z_c_mu_hat = infer_ans['z_c_mu_hat']
        z_c_logsigma_hat = infer_ans['z_c_logsigma_hat']
        z_c_broad = z_c.unsqueeze(1).repeat(1, self.k, 1)
        j = torch.argmin(((z_c_logsigma_hat.exp() - z_c_broad) ** 2).sum(2), dim=1)
        j = j.unsqueeze(1).unsqueeze(1).repeat(1, 1, 256)
        z_c_mu_hat_j = torch.gather(z_c_mu_hat, 1, j).squeeze()
        z_c_logsigma_hat_j = torch.gather(z_c_logsigma_hat, 1, j).squeeze()
        z_c_hat = z_c_mu_hat_j + torch.randn_like(z_c_logsigma_hat_j) * z_c_logsigma_hat_j.exp()
        
        return z_c_hat
    
    def forward(self):
        """Ours forward"""
        # encoder process
        z, z_mu, z_logsigma, f = self.net_E(self.img_m, self.img_c)

        z_m_mu = torch.chunk(z_mu, 2, dim=0)[0]
        z_m_logsigma = torch.chunk(z_logsigma, 2, dim=0)[0]
        z_c_mu = torch.chunk(z_mu, 2, dim=0)[1]
        z_c_logsigma = torch.chunk(z_logsigma, 2, dim=0)[1]

        # decoder process
        z_m, z_c, f_m, f_e, mask = self.get_G_inputs(z, f)
        z_c_hat = self.get_z_c_hat(z_m, z_c)
        z = torch.cat([z_m, z_c], dim=1)
        if self.isTrain:
            z_hat = torch.cat([z_m, z_c_hat], dim=1)
        else:
            z_hat = torch.cat([z_m, z_c_hat.unsqueeze(0)], dim=1)
        results, attn = self.net_G(z, f_m, f_e, mask)
        results_hat, attn_hat = self.net_G(z_hat, f_m, f_e, mask)
        self.img_g = []
        self.img_g_hat = []
        for result in results:
            self.img_g.append(result)
        for result_hat in results_hat:
            self.img_g_hat.append(result_hat)
        self.img_out = (1-self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth
        self.img_out_hat = (1-self.mask) * self.img_g_hat[-1].detach() + self.mask * self.img_truth
        self.z_m = z_m
        self.z_c = z_c
        self.z_c_mu = z_c_mu
        self.z_c_logsigma = z_c_logsigma
        self.z_m_mu = z_m_mu
        self.z_m_logsigma = z_m_logsigma

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss +=gradient_penalty

        D_loss.backward()

        return D_loss
    
    def backward_D(self):
        base_function._unfreeze(self.net_D)
        # self.loss_ad_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        loss_ad_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        loss_ad_d = loss_ad_d + self.backward_D_basic(self.net_D, self.img_truth, self.img_g_hat[-1])
        self.loss_ad_d = loss_ad_d
    
    def backward_G(self):
        base_function._unfreeze(self.net_D)
        # g loss fake
        D_fake = self.net_D(self.img_g[-1])
        loss_ad_g = self.GANloss(D_fake, True, False)

        # gmm loss fake
        D_fake_hat = self.net_D(self.img_g_hat[-1])
        D_real_hat = self.net_D(self.img_truth)
        loss_ad_g = loss_ad_g + self.L2loss(D_fake_hat, D_real_hat)
        self.loss_ad_g = loss_ad_g * self.opt.lambda_g

        # l1 loss
        loss_rec = 0
        for i, (img_fake_i, img_fake_hat_i, img_real_i, mask_i) in enumerate(zip(self.img_g, self.img_g_hat, self.scale_img, self.scale_mask)):
            loss_rec += self.L1loss(img_fake_hat_i * mask_i, img_real_i * mask_i)
            loss_rec += self.L1loss(img_fake_i, img_real_i)
        self.loss_rec = loss_rec * self.opt.lambda_rec

        # elbo loss
        # p_distribution = torch.distributions.Normal(torch.zeros_like(self.z_m_mu), torch.ones_like(self.z_m_logsigma))
        # q_distribution = torch.distributions.Normal(self.z_m_mu, self.z_m_logsigma.exp())
        # loss_elbo = torch.distributions.kl_divergence(p_distribution, q_distribution)
        # self.loss_elbo = loss_elbo.mean() * self.opt.lambda_kl

        total_loss = 0
        for name in self.loss_names:
            if name != 'ad_d' and name != 'freq' and name != 'bm':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()
    
    def backward_GMM(self):
        base_function._unfreeze(self.net_GMM)

        infer_ans = self.net_GMM(self.z_m, self.z_c)
        alpha = infer_ans['alpha']
        z_c_mu_hat = infer_ans['z_c_mu_hat']
        z_c_logsigma_hat = infer_ans['z_c_logsigma_hat']
        z_c_broad = self.z_c.unsqueeze(1).repeat(1, self.k, 1)
        j = torch.argmin(((z_c_logsigma_hat.exp() - z_c_broad) ** 2).sum(2), dim=1)
        v = F.one_hot(j, self.k)
        j = j.unsqueeze(1).unsqueeze(1).repeat(1, 1, 256)
        self.loss_freq = self.L2loss(alpha, v)

        z_c_mu_hat_j = torch.gather(z_c_mu_hat, 1, j).squeeze()
        z_c_logsigma_hat_j = torch.gather(z_c_logsigma_hat, 1, j).squeeze()
        z_c_mu = self.z_c_mu.detach()
        z_c_logsigma = self.z_c_logsigma.detach()

        self.loss_bm = - (1 / 2) * ((z_c_logsigma_hat_j - z_c_logsigma) 
                                    - ((1 / z_c_logsigma.exp()) * z_c_logsigma_hat_j.exp()) 
                                    + (z_c_mu_hat_j - z_c_mu) ** 2 * (1 / z_c_logsigma.exp())).sum(1).mean()

        loss = self.loss_freq + self.loss_bm 
        loss = torch.tensor(loss, dtype=float)
        loss.requires_grad_()

        loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # optimize gmm parameters
        self.optimizer_GMM.zero_grad()
        self.backward_GMM()
        self.optimizer_GMM.step()
