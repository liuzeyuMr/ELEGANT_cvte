from dataset import config, MultiCelebADataset
from nets import Encoder, Decoder, Discriminator
# import sys
# sys.path.append('../')
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from itertools import chain


class ELEGANT(object):
    def __init__(self, args,
                 config=config, dataset=MultiCelebADataset, \
                 encoder=Encoder, decoder=Decoder, discriminator=Discriminator):

        self.args = args
        self.attributes = args.attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#判断是否可用cuda

        self.n_attributes = len(self.attributes) #选择属性的个数
        self.gpu = args.gpu
        self.mode = args.mode #train or test
        self.restore = args.restore  #加载模型index

        # init dataset and networks
        self.config = config
        self.dataset = dataset(self.attributes)
        self.Enc = encoder()
        self.Dec = decoder()
        self.D1  = discriminator(self.n_attributes, self.config.nchw[-1])#256 #这里定义了两个判别器
        self.D2  = discriminator(self.n_attributes, self.config.nchw[-1]//2)#128

        self.adv_criterion = torch.nn.BCELoss()# Binary cross entropy
        self.recon_criterion = torch.nn.MSELoss()
        self.MSE_loss=torch.nn.MSELoss()
        self._criterion_cycle = torch.nn.L1Loss().cuda()
        self._criterion_D_cond = torch.nn.MSELoss().cuda()

        self.restore_from_file() #从现有的文件中恢复权重参数
        self.set_mode_and_gpu()  #


    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.restore))#获得编码网络地址
            assert os.path.exists(ckpt_file_enc)
            ckpt_file_dec = os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.restore))#获得解码网络地址
            assert os.path.exists(ckpt_file_dec)
            if self.gpu:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)#恢复生成网络的参数
                self.Dec.load_state_dict(torch.load(ckpt_file_dec), strict=False)
            else:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc, map_location='cpu'), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec, map_location='cpu'), strict=False)

            if self.mode == 'train': #如果是训练，才恢复已有的判别网络参数
                ckpt_file_d1  = os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d1)
                ckpt_file_d2  = os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.restore))
                assert os.path.exists(ckpt_file_d2)
                if self.gpu:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2), strict=False)
                else:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1, map_location='cpu'), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2, map_location='cpu'), strict=False)

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.Enc.train() #训练模式
            self.Dec.train()
            self.D1.train()
            self.D2.train()

            self.writer = SummaryWriter(self.config.log_dir)
            #优化生成网络
            self.optimizer_G = torch.optim.Adam(chain(self.Enc.parameters(), self.Dec.parameters()),
                                           lr=self.config.G_lr, betas=(0.5, 0.999),
                                           weight_decay=self.config.weight_decay)
            #优化判别网络
            self.optimizer_D = torch.optim.Adam(chain(self.D1.parameters(),self.D2.parameters()),
                                           lr=self.config.D_lr, betas=(0.5, 0.999),
                                           weight_decay=self.config.weight_decay)



            self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.step_size, gamma=self.config.gamma)#学习率衰减
            self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.step_size, gamma=self.config.gamma)
            if self.restore is not None:
                for _ in range(self.restore):
                    self.D_lr_scheduler.step()
                    self.D_lr_scheduler.step()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()
                    self.D1.cuda()
                    self.D2.cuda()
                    self.adv_criterion.cuda()
                    self.recon_criterion.cuda()
                    self.MSE_loss.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))
                self.D1  = torch.nn.DataParallel(self.D1, device_ids=list(range(len(self.gpu))))
                self.D2  = torch.nn.DataParallel(self.D2, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'test':
            self.Enc.eval()#让model变成测试模型
            self.Dec.eval()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda() #测试只需要生产网络
                    self.Dec.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))

        else:
            raise NotImplementationError()

    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            var = torch.autograd.Variable(tensor, volatile=volatile)
            out.append(var)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        num_chs = encodings.size(1) #512维的通道数
        per_chs = float(num_chs) / self.n_attributes #eg.n_attributes=4 将128通道当成一个属性的代表通道
        start = int(np.rint(per_chs * attribute_id))#确定开始的通道数
        end = int(np.rint(per_chs * (attribute_id + 1)))#确定最后的通道数
        # return encodings[:, start:end]
        return encodings.narrow(1, start, end-start) #表示取出tensor中第1维上索引从start开始到end-start的所有元素

    def forward_G(self):
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)#输入图片，获得编码向量和特征集 表示有属性
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)#输入图片，获得编码向量和特征集 表示无属性
        # print(self.A.shape)
        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_B, i)  for i in range(self.n_attributes)], 1)
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)

        # self.A_fake_imgs, self.A_fake_img_mask = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
        # self.B_fake_imgs, self.B_fake_img_mask = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
        # self.C_fake_imgs, self.C_fake_img_mask = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
        # self.D_fake_imgs, self.D_fake_img_mask = self.Dec(self.z_D, self.z_B, skip=self.B_skip)

        self.A_fake_img_mask = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
        self.B_fake_img_mask = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
        self.C_fake_img_mask = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
        self.D_fake_img_mask = self.Dec(self.z_D, self.z_B, skip=self.B_skip)
        # print(A_fake_img_mask.shape)   torch.Size([16, 1, 256, 256])
        # print(self.A)


        # self.A1 = self.A_fake_img_mask * self.A + (1 - self.A_fake_img_mask) * self.A_fake_imgs #进行重构 A'
        # self.B1 = self.B_fake_img_mask * self.B + (1 - self.B_fake_img_mask) * self.B_fake_imgs #重构     B'
        # self.C  = self.C_fake_img_mask * self.A + (1 - self.C_fake_img_mask) * self.C_fake_imgs #无属性   C
        # self.D  = self.D_fake_img_mask * self.B + (1 - self.D_fake_img_mask) * self.D_fake_imgs #有属性   D

        self.A1 = torch.clamp(self.A + self.A_fake_img_mask, -1, 1)
        self.B1 = torch.clamp(self.B + self.B_fake_img_mask, -1, 1)
        self.C = torch.clamp(self.A + self.C_fake_img_mask, -1, 1)
        self.D = torch.clamp(self.B + self.D_fake_img_mask, -1, 1)

    def forward_D_real_sample(self):
        self.d1_A_prob ,self.d1_A_cond = self.D1(self.A) #真实图像和标签 256*256 因为有两个判别网络所以有四个值
        self.d1_B_prob ,self.d1_B_cond = self.D1(self.B)
        self.d2_A_prob ,self.d2_A_cond=  self.D2(self.A)#真实图像和标签 128*128
        self.d2_B_prob ,self.d2_B_cond=  self.D2(self.B)
        # print(self.y_A)
        # print(self.y_B)
        # print(self.MSE_loss(self.d1_A_cond,self.y_B))
        # print(torch.mean(self.d2_A_prob))
        # print(torch.mean(self.y_A))


    def forward_D_fake_sample(self, detach):
        self.y_C, self.y_D = self.y_A.clone(), self.y_B.clone()
        self.y_C.data[:, self.attribute_id] = self.y_B.data[:, self.attribute_id]#将属性进行更改，将B的某一属性给A得到C
        self.y_D.data[:, self.attribute_id] = self.y_A.data[:, self.attribute_id]#将A的某一属性给B得到D

        if detach: #梯度不会顺着到G网络
            self.d1_C_prob , _ = self.D1(self.C.detach())#得到假的图像的判别值，这里只更新D网络
            self.d1_D_prob , _ = self.D1(self.D.detach())
            self.d2_C_prob , _ = self.D2(self.C.detach())
            self.d2_D_prob , _ = self.D2(self.D.detach())
        else: #用来更新G网络
            self.d1_C_prob, self.d1_C_cond = self.D1(self.C)  # 得到假的图像的判别值，这里只更新G网络
            self.d1_D_prob, self.d1_D_cond = self.D1(self.D)
            self.d2_C_prob, self.d2_C_cond = self.D2(self.C)
            self.d2_D_prob, self.d2_D_cond = self.D2(self.D)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def compute_gp(self,A,C,n):
        alpha = torch.rand(A.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * A.data + (1 - alpha) * C.data).requires_grad_(True)

        if n==1:#计算D1
            out_src1 , _ = self.D1(x_hat)  # 求出interploated的分布

            d_loss_gp = self.gradient_penalty(out_src1, x_hat)  # 求出分布和中间之间的梯度
        else:   #计算D2
            out_src1 , _ = self.D2(x_hat)  # 求出interploated的分布
            d_loss_gp = self.gradient_penalty(out_src1, x_hat)  # 求出分布和中间之间的梯度
        return  d_loss_gp


    def compute_loss_D(self):
        # Compute loss for gradient penalty.
        # alpha1 = torch.rand(self.A.size(0), 1, 1, 1).to(self.device)
        # x_hat1 = (alpha1 * self.A.data + (1 - alpha1) * self.C.data).requires_grad_(True)
        # x_hat2 = (alpha1 * self.d1_B.data + (1 - alpha1) * self.d1_D.data).requires_grad_(True)
        #
        # out_src1, _ = self.D1(x_hat1,self.y_A)  # 求出interploated的分布
        # d_loss_gp1 = self.gradient_penalty(out_src1, x_hat1)  # 求出分布和中间之间的梯度
        #
        # out_src2, _ = self.D1(x_hat2, self.y_B)  # 求出interploated的分布
        # d_loss_gp2 = self.gradient_penalty(out_src2, x_hat2)  # 求出分布和中间之间的梯度

        self.D_loss = {
            # 这里衡量的是JS散度
            # 'D1':   self.adv_criterion(self.d1_A, torch.ones_like(self.d1_A))  + \
            #         self.adv_criterion(self.d1_B, torch.ones_like(self.d1_B))  + \
            #         self.adv_criterion(self.d1_C, torch.zeros_like(self.d1_C)) + \
            #         self.adv_criterion(self.d1_D, torch.zeros_like(self.d1_D)),
            #
            # 'D2':   self.adv_criterion(self.d2_A, torch.ones_like(self.d2_A))  + \
            #         self.adv_criterion(self.d2_B, torch.ones_like(self.d2_B))  + \
            #         self.adv_criterion(self.d2_C, torch.zeros_like(self.d2_C)) + \
            #         self.adv_criterion(self.d2_D, torch.zeros_like(self.d2_D)),


            'D1': -torch.mean(self.d1_A_prob) + torch.mean(self.d1_C_prob)+10.0*self.compute_gp(self.A,self.C,1)\
                  -torch.mean(self.d1_B_prob) + torch.mean(self.d1_D_prob)+10.0*self.compute_gp(self.B,self.D,1),

            'D2': -torch.mean(self.d2_A_prob) + torch.mean(self.d2_C_prob)+10.0*self.compute_gp(self.A,self.C,0)\
                  -torch.mean(self.d2_B_prob) + torch.mean(self.d2_D_prob)+10.0*self.compute_gp(self.B,self.D,0),

            'condD1': self.MSE_loss(self.d1_A_cond, self.y_A) *0.01 + self.MSE_loss(self.d1_B_cond,self.y_B)*0.01,

            'condD2': self.MSE_loss(self.d2_A_cond, self.y_A) *0.01 + self.MSE_loss(self.d2_B_cond,self.y_B)*0.01,

            # 'D1': -torch.mean(self.d1_A) + torch.mean(self.d1_C)  \
            #       - torch.mean(self.d1_B) + torch.mean(self.d1_D) ,
            #
            # 'D2': -torch.mean(self.d2_A) + torch.mean(self.d2_C)  \
            #       - torch.mean(self.d2_B) + torch.mean(self.d2_D) ,

            # 'D1': self.MSE_loss(self.d1_A, torch.ones_like(self.d1_A)) + \
            #       self.MSE_loss(self.d1_B, torch.ones_like(self.d1_B)) + \
            #       self.MSE_loss(self.d1_C, torch.zeros_like(self.d1_C)) + \
            #       self.MSE_loss(self.d1_D, torch.zeros_like(self.d1_D)),
            #
            # 'D2': self.MSE_loss(self.d2_A, torch.ones_like(self.d2_A)) + \
            #       self.MSE_loss(self.d2_B, torch.ones_like(self.d2_B)) + \
            #       self.MSE_loss(self.d2_C, torch.zeros_like(self.d2_C)) + \
            #       self.MSE_loss(self.d2_D, torch.zeros_like(self.d2_D)),

        }
        self.loss_D = (self.D_loss['D1'] + 0.5 * self.D_loss['D2']+self.D_loss['condD1']+self.D_loss['condD2']) / 4

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def compute_loss_G(self):
        self.G_loss = {
            'reconstruction': self.recon_criterion(self.A1, self.A) + self.recon_criterion(self.B1, self.B), #重构误差


            # 'adv1': self.adv_criterion(self.d1_A_prob, torch.ones_like(self.d1_A_prob)) + \
            #         self.adv_criterion(self.d1_A_prob, torch.ones_like(self.d1_A_prob)),
            # 'adv2': self.adv_criterion(self.d1_A_prob, torch.ones_like(self.d1_A_prob)) + \
            #         self.adv_criterion(self.d1_A_prob, torch.ones_like(self.d1_A_prob)),


            'adv1': -torch.mean(self.d1_C_prob)-torch.mean(self.d1_D_prob),
            'adv2': -torch.mean(self.d2_C_prob)-torch.mean(self.d1_D_prob),

            'g_mask':torch.mean(self.A_fake_img_mask) + torch.mean(self.B_fake_img_mask)+\
                     torch.mean(self.C_fake_img_mask) + torch.mean(self.D_fake_img_mask),

            'g_mask_smooth':self._compute_loss_smooth(self.A_fake_img_mask)+self._compute_loss_smooth(self.B_fake_img_mask)+ \
                            self._compute_loss_smooth(self.C_fake_img_mask)+self._compute_loss_smooth(self.D_fake_img_mask),

        #     # loss mask
        #     self._loss_g_mask_1 = torch.mean(fake_img_mask) * self._opt.lambda_mask
        # self._loss_g_mask_2 = torch.mean(rec_real_img_mask) * self._opt.lambda_mask
        # self._loss_g_mask_1_smooth = self._compute_loss_smooth(fake_img_mask) * self._opt.lambda_mask_smooth
        # self._loss_g_mask_2_smooth = self._compute_loss_smooth(rec_real_img_mask) * self._opt.lambda_mask_smooth


            #
            # 'adv1': self.MSE_loss(self.d1_C, torch.ones_like(self.d1_C)) + \
            #         self.MSE_loss(self.d1_D, torch.ones_like(self.d1_D)),
            # 'adv2': self.MSE_loss(self.d2_C, torch.ones_like(self.d2_C)) + \
            #         self.MSE_loss(self.d2_D, torch.ones_like(self.d2_D)),
        }
        # self.loss_G = 5 * self.G_loss['reconstruction'] + self.G_loss['adv1'] + 0.5 * self.G_loss['adv2']+10 * self.G_loss['g_mask']+\
        #               0.001 * self.G_loss['g_mask_smooth']
        self.loss_G = 5 * self.G_loss['reconstruction'] + self.G_loss['adv1'] + 0.5 * self.G_loss['adv2']


    def backward_D(self):
        self.loss_D.backward()
        self.optimizer_D.step()#更新判别网络参数
        # for p in self.D1.parameters():
        #     p.data.clamp_(-0.01, 0.01)
        # for q in self.D2.parameters():
        #     q.data.clamp_(-0.01, 0.01)

    def backward_G(self):
        self.loss_G.backward()
        self.optimizer_G.step()

    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def save_image_log(self, save_num=20):
        image_info = {
            'A/img'   : self.img_denorm(self.A.data.cpu(), 1)[:save_num],
            'B/img'   : self.img_denorm(self.B.data.cpu(), 1)[:save_num],
            'C/img'   : self.img_denorm(self.C.data.cpu(), 1)[:save_num],
            'D/img'   : self.img_denorm(self.D.data.cpu(), 1)[:save_num],
            'A1/img'  : self.img_denorm(self.A1.data.cpu(), 1)[:save_num],
            'B1/img'  : self.img_denorm(self.B1.data.cpu(), 1)[:save_num],
            'R_A/img' : self.img_denorm(self.A_fake_img_mask.data.cpu(), 1)[:save_num],
            'R_B/img' : self.img_denorm(self.B_fake_img_mask.data.cpu(), 1)[:save_num],
            'R_C/img' : self.img_denorm(self.C_fake_img_mask.data.cpu(), 1)[:save_num],
            'R_D/img' : self.img_denorm(self.D_fake_img_mask.data.cpu(), 1)[:save_num],
        }
        for tag, images in image_info.items():
            for idx, image in enumerate(images):
                self.writer.add_image(tag+'/{}_{:02d}'.format(self.attribute_id, idx), image, self.step)

    def save_sample_images(self, save_num=5):
        # canvas = torch.cat((self.A, self.B, self.C, self.D, self.A1, self.B1), -1)
        # img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)
        # for i in range(save_num):
        #     Image.fromarray(img_array[i]).save(os.path.join(self.config.img_dir, 'step_{:06d}_attr_{}_{:02d}.jpg'.format(self.step, self.attribute_id, i)))
        #
        self.D_fake_img_mask = torch.cat((self.D_fake_img_mask, self.D_fake_img_mask, self.D_fake_img_mask), 1)
        self.C_fake_img_mask = torch.cat((self.C_fake_img_mask, self.C_fake_img_mask, self.C_fake_img_mask),1)

        # print(self.C_fake_img_mask.shape)
        # print(self.C_fake_imgs.shape)
        canvas = torch.cat((self.A, self.B, self.C, self.D, self.C_fake_img_mask, self.D_fake_img_mask), -1)
        img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()), (0, 2, 3, 1)).astype(np.uint8)
        for i in range(save_num):
            Image.fromarray(img_array[i]).save(os.path.join(self.config.img_dir,
                                                            'step_{:06d}_attr_{}_{:02d}.jpg'.format(self.step,
                                                                                                    self.attribute_id,
                                                                                                    i)))
    def save_scalar_log(self):
        scalar_info = {
            'loss_D': self.loss_D.data.cpu().numpy(),
            'loss_G': self.loss_G.data.cpu().numpy(),
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0],
        }

        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value.item()

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value.item()

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, self.step)

    def save_model(self):
        reduced = lambda key: key[7:] if key.startswith('module.') else key
        torch.save({reduced(key): val.cpu() for key, val in self.Enc.state_dict().items()}, os.path.join(self.config.model_dir, 'Enc_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.Dec.state_dict().items()}, os.path.join(self.config.model_dir, 'Dec_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D1.state_dict().items()},  os.path.join(self.config.model_dir, 'D1_iter_{:06d}.pth'.format(self.step)))
        torch.save({reduced(key): val.cpu() for key, val in self.D2.state_dict().items()},  os.path.join(self.config.model_dir, 'D2_iter_{:06d}.pth'.format(self.step)))

    def train(self):
        for self.step in range(self.start_step, 1 + self.config.max_iter):
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            for self.attribute_id in range(self.n_attributes): # 0 1
                #这里是得到属性 并分别获得有这个属性和没有这个属性的数据集
                A, y_A = next(self.dataset.gen(self.attribute_id, True))
                B, y_B = next(self.dataset.gen(self.attribute_id, False))
                # print(y_A)
                # print(type(A))
                self.A, self.y_A, self.B, self.y_B = self.tensor2var([A, y_A, B, y_B])#将tensor变成variable
                # print(y_A)
                # print(y_B)
                # print(self.y_A)


                # forward
                self.forward_G()

                # update D 更新判别网络
                self.forward_D_real_sample() #将真实图像输入到判别网络 只更新判别网络
                self.forward_D_fake_sample(detach=True) #将G网络生成的fake_image输入判别网络，detach=True 只更新判别网络
                self.compute_loss_D() #计算D_loss
                self.optimizer_D.zero_grad()
                self.backward_D()

                # update G
                self.forward_D_fake_sample(detach=False)
                self.compute_loss_G()
                self.optimizer_G.zero_grad()
                self.backward_G()

                if self.step % 1000 == 0:#1000步保存一次日志参数
                    self.save_image_log()

                if self.step % 200 == 0:# 1000步保存一次测试图片
                    self.save_sample_images()

            print('step: %06d, loss D: %.6f, loss G: %.6f' % (self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy()))

            if self.step % 1000 == 0:
                self.save_scalar_log()

            if self.step % 2000 == 0:
                self.save_model()

        print('Finished Training!')
        self.writer.close()

    def transform(self, *images):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1
        out = [transform2(transform1(image)) for image in images]
        return out

    def swap(self):
        '''
        swap attributes of two images.
        '''
        self.attribute_id = self.args.swap_list[0]
        self.B, self.A = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0])), volatile=True)

        self.forward_G()
        img = torch.cat((self.B, self.A, self.D, self.C), -1)
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('swap.jpg')

    def linear(self):
        '''
        linear interpolation of two images.
        '''
        self.attribute_id = self.args.swap_list[0] #
        self.B, self.A = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0])), volatile=True)

        self.z_A = self.Enc(self.A, return_skip=False)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        #只插值一个属性
        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)

        m = self.args.size[0]
        out = [self.B]
        for i in range(1, 1+m):
            z_i = float(i) / m * (self.z_D - self.z_B) + self.z_B#当i==m时 z_i等于z_D
            R_i = self.Dec(z_i, self.z_B, skip=self.B_skip)
            D_i = torch.clamp(self.B + R_i, -1, 1)
            out.append(D_i)
        out.append(self.A)
        out = torch.cat(out, -1)
        img = np.transpose(self.img_denorm(out.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('linear_interpolation.jpg')

    def matrix1(self):
        '''
        matrix interpolation with respect to one attribute.
        '''
        self.attribute_id = self.args.swap_list[0]
        self.B = self.tensor2var(self.transform(Image.open(self.args.input)), volatile=True)
        self.As = [self.tensor2var(self.transform(Image.open(self.args.target[i])), volatile=True) for i in range(3)]

        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_As = [self.Enc(self.As[i], return_skip=False) for i in range(3)]

        self.z_Ds = [torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_As[j], i)  for i in range(self.n_attributes)], 1)
                     for j in range(3)]

        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        out = torch.ones(1, 3, m * h, n * w)
        for i in range(m):
            for j in range(n):
                a = i / float(m - 1)
                b = j / float(n - 1)
                four = [(1-a) * (1-b), (1-a) * b, a * (1-b), a * b]
                z_ij = four[0] * self.z_B + four[1] * self.z_Ds[0] + four[2] * self.z_Ds[1] + four[3] * self.z_Ds[2]
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)
                out[:,:, i*h:(i+1)*h, j*w:(j+1)*w] = D_ij.data.cpu()

        first_col = torch.cat((self.B.data.cpu(), torch.ones(1,3,(m-2)*h,w), self.As[1].data.cpu()), -2)
        last_col = torch.cat((self.As[0].data.cpu(), torch.ones(1,3,(m-2)*h,w), self.As[2].data.cpu()), -2)
        canvas = torch.cat((first_col, out, last_col), -1)
        img = np.transpose(self.img_denorm(canvas.numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('matrix_interpolation1.jpg')

    def matrix2(self):
        '''
        matrix interpolation with respect to two attributes simultaneously.
        '''
        self.attribute_ids = self.args.swap_list
        self.B, self.A1, self.A2 = self.tensor2var(self.transform(Image.open(self.args.input), Image.open(self.args.target[0]), Image.open(self.args.target[1])), volatile=True)

        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)
        self.z_A1, self.z_A2 = self.Enc(self.A1, return_skip=False), self.Enc(self.A2, return_skip=False)

        self.z_D1 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[0]
                              else self.get_attr_chs(self.z_A1, i)  for i in range(self.n_attributes)], 1)

        self.z_D2 = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_ids[1]
                              else self.get_attr_chs(self.z_A2, i)  for i in range(self.n_attributes)], 1)

        m, n = self.args.size
        h, w = self.config.nchw[-2:]

        out = torch.ones(1, 3, m * h, n * w)
        for i in range(m):
            for j in range(n):
                a = i / float(m - 1)
                b = j / float(n - 1)
                z_ij = a * self.z_D1 + b * self.z_D2 + (1 - a - b) * self.z_B
                R_ij = self.Dec(z_ij, self.z_B, skip=self.B_skip)
                D_ij = torch.clamp(self.B + R_ij, -1, 1)
                out[:,:, i*h:(i+1)*h, j*w:(j+1)*w] = D_ij.data.cpu()

        first_col = torch.cat((self.B.data.cpu(), torch.ones(1,3,(m-2)*h,w), self.A1.data.cpu()), -2)
        last_col = torch.cat((self.A2.data.cpu(), torch.ones(1,3,(m-1)*h,w)), -2)
        canvas = torch.cat((first_col, out, last_col), -1)
        img = np.transpose(self.img_denorm(canvas.numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save('matrix_interpolation2.jpg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attributes', nargs='+', default=['Bangs','Smiling'],type=str, help='Specify attribute names.')#属性
    parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=str, help='Specify GPU ids.') #gpu 测试的时候需要设置一个
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test']) #
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore')

    # test parameters
    parser.add_argument('--swap', action='store_true', help='Swap attributes.')
    parser.add_argument('--linear', action='store_true', help='Linear interpolation.')
    parser.add_argument('--matrix', action='store_true', help='Matraix interpolation with respect to one attribute.')
    parser.add_argument('--swap_list', default=[], nargs='+', type=int, help='Specify the attributes ids for swapping.')
    parser.add_argument('-i', '--input', type=str, help='Specify the input image.')
    parser.add_argument('-t', '--target', nargs='+', type=str, help='Specify target images.')
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Specify the interpolation size.')

    args = parser.parse_args()
    print(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    if args.mode == 'test':
        assert args.swap + args.linear + args.matrix == 1
        assert args.restore is not None

    model = ELEGANT(args) # 获得模型对象
    if args.mode == 'train':
        model.train()#训练
    elif args.mode == 'test' and args.swap:#
        assert len(args.swap_list) == 1 and args.input and len(args.target) == 1
        model.swap()
    elif args.mode == 'test' and args.linear:
        assert len(args.swap_list) == 1 and len(args.size) == 1
        model.linear()
    elif args.mode == 'test' and args.matrix:
        assert len(args.swap_list) in [1,2]
        if len(args.swap_list) == 1:#只交换一个属性
            assert len(args.target) == 3 and len(args.size) == 2
            model.matrix1()
        elif len(args.swap_list) == 2:#交换两个属性
            assert len(args.target) == 2 and len(args.size) == 2
            model.matrix2()
    else:
        raise NotImplementationError()


if __name__ == "__main__":
    main()
