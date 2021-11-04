import torch
from collections import OrderedDict
from torch.nn import utils, DataParallel, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from model import build_model, weights_init
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter

EPSILON = 2.2204e-16
p = OrderedDict()

base_model_cfg = 'resnet' #vgg or resnet
p['lr_bone'] = 3e-5  # resnet 3e-5 vgg 2e-5
p['backbone'] = 1e-5 # resnet 1e-5 vgg 2e-5
p['wd'] = 0.0005  # weight decay
p['momentum'] = 0.90
lr_decay_epoch = [] #VGG-30 ResNet-none
nAveGrad = 10  # update the weights once in 'nAveGrad' forward passes
showEvery = 50


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.config = config
        self.save_fold = save_fold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = (torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1)).to(self.device)
        self.build_model()
        if self.config.pre_trained:
            self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            print('Training')
            # self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)  # location of the trained model
            self.net_bone.load_state_dict(torch.load(self.config.model), strict=False)
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        if self.config.cuda:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.net_bone = DataParallel(self.net_bone)
            self.net_bone.to(self.device)
            self.net_bone.train()
            self.net_bone.apply(weights_init)
            if self.config.mode == 'train':
                if self.config.load_bone == '':
                    if base_model_cfg == 'vgg':
                        self.net_bone.base.base.load_state_dict(torch.load(self.config.vgg))
                    elif base_model_cfg == 'resnet':
                        self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
                if self.config.load_bone != '':
                    self.net_bone.load_state_dict(torch.load(self.config.load_bone), strict=False)

            self.backbone = list(map(id, self.net_bone.base.parameters()))
            self.bone = filter(lambda p: id(p) not in self.backbone, self.net_bone.parameters())

        self.lr_bone = p['lr_bone']
        self.lr_backbone = p['backbone']
        self.weight_decay = p['wd']
        self.momentum = p['momentum']
        self.optimizer_bone = Adam(
            [
                {'params': self.bone, 'lr': self.lr_bone},
                {'params': self.net_bone.base.parameters(), 'lr': self.lr_backbone},
            ], weight_decay=self.weight_decay)

        self.print_network(self.net_bone, 'trueUnify bone part')

    def test(self, test_mode=0):
        img_num = len(self.test_loader)
        time_t = 0.0
        name_t = self.config.modelname + '/'

        if not os.path.exists(os.path.join(self.save_fold, name_t)):
            os.makedirs(os.path.join(self.save_fold, name_t))
        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            images_, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images_)
                if self.config.cuda:
                    images = images.to(self.device)
                torch.cuda.synchronize()
                time_start = time.time()
                final_sal, up_sal, edge_lossreturn = self.net_bone(images)
                time_end = time.time()
                print(time_end - time_start)
                time_t = time_t + time_end - time_start
                pred = np.squeeze(torch.sigmoid(final_sal[-1]).cpu().data.numpy())  # from variable to numpy
                # pred = np.squeeze(torch.sigmoid(NLB[0]).cpu().data.numpy())
                multi_fuse = 255 * pred
                print(pred.shape)
                cv2.imwrite(os.path.join(self.config.test_fold, name_t, name[:-4] + '.png'), multi_fuse)
                print(os.path.join(self.config.test_fold, name_t, name[:-4] + '.png'))
        print("--- %s seconds ---" % (time_t))
        print('Test Done!')

    def train(self):
        aveGrad = 0

        writer = SummaryWriter(log_dir='tensorboard/' + self.config.modelname)
        for epoch in range(self.config.epoch):
            l1, l2, r_sal_loss = 0, 0, 0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']
                sal_image = sal_image.to(self.device)
                sal_label = sal_label.to(self.device)
                sal_edge = sal_edge.to(self.device)

                if sal_image.size()[2:] != sal_label.size()[2:]:
                    print("Skip this batch")
                    continue

                final_sal, up_sal, edge_lossreturn = self.net_bone(sal_image)

                # sal part
                sal_loss1 = []
                sal_loss2 = []
                sal_loss3 = []

                for ix in up_sal:
                    sal_loss1.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in final_sal:
                    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in edge_lossreturn:
                    sal_loss3.append(bce2d_new(ix, sal_edge, reduction='sum'))

                sal_loss = (sum(sal_loss1) + sum(sal_loss2) + sum(sal_loss3)) / (nAveGrad * self.config.batch_size)

                l1 = sum(sal_loss1).data
                l2 = sum(sal_loss2).data
                l3 = sum(sal_loss3).data
                r_sal_loss += sal_loss.data
                loss = sal_loss
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                if i % showEvery == 0:

                    writer.add_scalar('Train/Each_SAL_LOSS', l1 * (nAveGrad * self.config.batch_size) / showEvery,
                                      i + epoch * len(self.train_loader.dataset))

                    writer.add_scalar('Train/Final_SAL', l2 * (nAveGrad * self.config.batch_size) / showEvery,
                                      i + epoch * len(self.train_loader.dataset))

                    writer.add_scalar('Train/EdgeLoss', l3 * (nAveGrad * self.config.batch_size) / showEvery,
                                      i + epoch * len(self.train_loader.dataset))

                    writer.add_scalar('Train/SUM_LOSS',
                                      r_sal_loss * (nAveGrad * self.config.batch_size) / showEvery,
                                      i + epoch * len(self.train_loader.dataset))

                    print('Learning rate: ' + str(self.lr_bone))
                    l1, l2, r_sal_loss, l3 = 0, 0, 0, 0

                # if i % 200 == 0:
                #     vutils.save_image(torch.sigmoid(final_sal[0]), '%s%s/tmp_salmap/epoch%d-iter%d-sal-0.jpg' % (self.config.save_fold, self.config.modelname, epoch, i), normalize=True, padding=0)
                #     vutils.save_image(torch.sigmoid(edge_lossreturn[-1]), '%s%s/tmp_salmap/epoch%d-iter%d-sal-0Edge.jpg' % (self.config.save_fold, self.config.modelname, epoch, i), normalize=True, padding=0)
                #     vutils.save_image(sal_image / 255. + self.mean / 255., '%s%s/tmp_salmap/epoch%d-iter%d-sal-data.jpg' % (self.config.save_fold, self.config.modelname, epoch, i), padding=0)
                #     vutils.save_image(sal_label, '%s%s/tmp_salmap/epoch%d-iter%d-sal-target.jpg' % (self.config.save_fold, self.config.modelname, epoch, i), padding=0)

            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1
                self.lr_backbone = self.lr_backbone * 0.1
                self.optimizer_bone = Adam(
                    [
                        {'params': self.bone, 'lr': self.lr_bone},
                        {'params': self.net_bone.base.parameters(), 'lr': self.lr_backbone},
                    ], weight_decay=self.weight_decay)
        torch.save(self.net_bone.state_dict(), '%s%s/models/final_bone.pth' % (self.config.save_fold, self.config.modelname))


def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()  # True False True False....
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
