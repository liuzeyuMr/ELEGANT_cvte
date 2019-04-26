# -*- coding:utf-8 -*-
# Created Time: Wed 07 Mar 2018 12:38:26 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class NTimesTanh(nn.Module):
    def __init__(self, N):
        super(NTimesTanh, self).__init__()
        self.N = N
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * self.N

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.alpha = Parameter(torch.ones(1))
        self.beta  = Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1)
        return x * self.alpha + self.beta

#编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3,64,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(64,128,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(128,256,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(256,512,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(512,512,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
        ])

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight, 1, 0.02)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.02)
                nn.init.constant(m.bias, 0)

    def forward(self, x, return_skip=True):
        skip = []#用来进行 skip connction
        for i in range(len(self.main)):
            x = self.main[i](x)
            if i < len(self.main) - 1:
                skip.append(x)#这里将特征层进行保留
        # print(len(skip))  4
        if return_skip:
            return x, skip
        else:
            return x

#解码器
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024,512,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512,256,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256,128,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128,64,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, bias=True),
            ),


        ])
        self.main2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32,3,3,1,1),
                nn.Tanh(),

            ),
            nn.Sequential(
                nn.Conv2d(32, 1, 3, 1, 1),
                nn.Sigmoid(),
            ),
        ])
        self.activation = NTimesTanh(2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight, 1, 0.02)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.02)
                nn.init.constant(m.bias, 0)

    def forward(self, enc1, enc2, skip=None):

        x = torch.cat([enc1, enc2], 1)
        for i in range(len(self.main)):  # len(self.main)=4     0 1 2 3
            x = self.main[i](x)
            if skip is not None and i < len(skip):
                x += skip[-i-1]
        # img_reg= self.main2[0](x)
        attention_reg = self.main2[1](x)

        return attention_reg

#判别网络
class Discriminator(nn.Module):
    def __init__(self, n_attributes, img_size):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes
        self.img_size = img_size
        self.conv1 = nn.Sequential(

            nn.Conv2d(3,64,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(64,128,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(128,256,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(256,512,3,2,1,bias=True),#这里输出通道数是512维
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

        )

        self.conv2 = nn.Sequential(
            nn.Linear(512*(self.img_size//16)*(self.img_size//16), 1),
            # nn.Sigmoid(), #衡量wgan不需要sigmoid
        )

        self.conv3 = nn.Sequential(
            nn.Linear(512 * (self.img_size // 16) * (self.img_size // 16), n_attributes),
            nn.Sigmoid(),
        )

        self.downsample = torch.nn.AvgPool2d(2, stride=2)

        # init weight 初始化权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight, 1, 0.02)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.02)
                nn.init.constant(m.bias, 0)

    def forward(self, image):
        '''
        image: (n * c * h * w)
        label: (n * n_attributes)
        '''
        while image.shape[-1] != self.img_size or image.shape[-2] != self.img_size: #如果不等于就直接切半
            image = self.downsample(image)
        # new_label = label.view((image.shape[0], self.n_attributes, 1, 1)).expand((image.shape[0], self.n_attributes, image.shape[2], image.shape[3]))
        # x = torch.cat([image, new_label], 1)

        h = self.conv1(image)#
        output = h.view(h.shape[0], -1)  # view和reshape

        out_real = self.conv2(output)
        out_aux = self.conv3(output)





        return out_real, out_aux

if __name__ == "__main__":
    enc = Encoder()
    dec = Decoder()
    D1 = Discriminator(3, 256)
    D2 = Discriminator(3, 128)

    imgs = Variable(torch.rand(32,3,256,256))
    labels = Variable(torch.ones(32,3))

    out, skip = enc(imgs)
    rec = dec(enc1=out, enc2=out, skip=skip)

    fake1 = D1(imgs, labels)
    fake2 = D2(imgs, labels)

    from IPython import embed; embed(); exit()
