import os
import math
from decimal import Decimal
import model
import loss
from option import args
from generate_mask import find_gradmask

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import sys
import numpy as np
import cv2 as cv
import imageio
import random
import copy
import torch.nn.functional as F
import torch.nn as nn

def grad_mask(model,args):
    params = model.named_parameters()
    mask = torch.load(args.dir_mask)

 
    with torch.no_grad():
        for name, g in params:
            key = name[6:]
            if 'weight' in key and 'mean' not in key:
               
                mask_w = mask[key]
               
                g.grad.data = g.grad.data * mask_w
               
    
                
def clip_grad(grad, clip_value = 1.0):
    for g in grad:
        g.data.clamp_(min=-clip_value, max=clip_value)

class Trainer_gradmask(nn.Module):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_gradmask, self ).__init__() 
        self.args = args
        self.scale = args.scale
        
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_val = loader.loader_val
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.args = args
        #self.inner_lr = 5e-5

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8


    
    def set_forward(self, lr , hr, num, epoch, is_feature = False):
        if int(epoch) == int(self.args.decay):
            self.args.inner_lr = self.args.inner_lr_decay

        lr_son = F.interpolate(lr, size=[int(lr.shape[2]//self.args.lr_son), int(lr.shape[3]//self.args.lr_son)], mode="bicubic", align_corners=True)
        sr = self.model(lr_son,0,num)
        loss = self.loss(sr,lr)

        trainable_value = {name:param for name, param in self.model.named_parameters() if 'mean' not in name}
        grad = torch.autograd.grad(loss, trainable_value.values())
        train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad,trainable_value.values(), trainable_value.keys())))
        for task_step in range(1,5):
            sr = self.model(lr_son,0,num,train_var)
            loss = self.loss(sr, lr) 
            grad = torch.autograd.grad(loss, train_var.values())
            train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad, train_var.values(), train_var.keys())))
        
        
        sr = self.model(lr,0,num, train_var)
        
        
        return sr

    
    def set_forward_loss(self, lr, hr, num, epoch):
        sr = self.set_forward(lr, hr, num, epoch, is_feature = False)
        loss = self.loss(sr, hr)

        return loss

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e} inner lr:{:.2e}'.format(epoch, Decimal(lr), Decimal(self.args.inner_lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)

        if epoch == 1:
            self.m0 = copy.deepcopy(self.model.state_dict())

        for batch, (lr, hr, num,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            if self.args.use_maml:
                loss_all = []
                avg_loss = 0
                count = 0
                for i in numlist:
                    if len(i[0])!=0:
                        loss = self.set_forward_loss(lr[i[0]], hr[i[0]], i[1], epoch)
                        avg_loss = avg_loss+loss.data
                        loss_all.append(loss)
                        count += 1
                avg_loss /= count
                
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
            else:
                sr = self.model(lr, 0, num)
                loss = self.loss(sr, hr)
                loss.backward()
                

            if self.args.grad_mask:
                grad_mask(self.model,args)
                

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            

            timer_model.hold()
            if self.args.use_maml:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t{:.1f}\t{:.1f}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        loss_q.data,
                        avg_loss.data,
                        timer_model.release(),
                        timer_data.release()))

                timer_data.tic()
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t{:.1f}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        loss.data,
                        timer_model.release(),
                        timer_data.release()))

                timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        if epoch == self.args.epochs - 1:
            m1 = self.m0
            m2 = self.model.state_dict()
            find_gradmask(m1,m2)


    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    filename[0] = filename[0].split('x')[0]
                    
                    if self.args.is45s:
                        if int(filename[0])<=150 or 1351 <= int(filename[0]) <= 1365:
                            flag = 0
                        elif 151 <= int(filename[0]) <= 300 or 1366 <= int(filename[0]) <= 1380:
                            flag = 1
                        elif 301 <= int(filename[0]) <= 450 or 1381 <= int(filename[0]) <= 1395:
                            flag = 2
                        elif 451 <= int(filename[0]) <= 600 or 1396 <= int(filename[0]) <= 1410:
                            flag = 3
                        elif 601 <= int(filename[0]) <= 750 or 1411 <= int(filename[0]) <= 1425:
                            flag = 4
                        elif 751 <= int(filename[0]) <= 900 or 1426 <= int(filename[0]) <= 1440:
                            flag = 5
                        elif 901 <= int(filename[0]) <= 1050 or 1441 <= int(filename[0]) <= 1455:
                            flag = 6
                        elif 1051 <= int(filename[0]) <= 1200 or 1456 <= int(filename[0]) <= 1470:
                            flag = 7
                        elif 1201 <= int(filename[0]) <= 1350 or 1471 <= int(filename[0]) <= 1485:
                            flag = 8
                        else:
                            flag = 9
                    
                    sr = self.model(lr, idx_scale, flag)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    
                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
            self.ckp.save_everyepoch(self, epoch, is_best=True)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

