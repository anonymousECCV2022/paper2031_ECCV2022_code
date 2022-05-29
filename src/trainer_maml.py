import os
import math
from decimal import Decimal
import model
import loss
from option import args
import pickle
import utility
from numba import jit
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
from model.patchnet import PatchNet
import csv
#from thop import profile, clever_format
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

class Trainer_maml(nn.Module):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_maml, self ).__init__() 
        
        self.args = args
        
        self.scale = args.scale
        self.psnr = []
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_val = loader.loader_val
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if self.args.patchnet:
            self.patchnet = PatchNet(args).to(self.device)


    def set_forward_attention(self, lr , hr, num, epoch, is_feature = False):
       
        if int(epoch) == int(self.args.decay.split('-')[0]):
            self.args.inner_lr = self.args.inner_lr_decay

       
        sr = self.model(lr,0,num)
        loss = self.loss(sr,hr)
       
        if self.args.patchnet:
            trainability = self.patchnet(sr,None)
            trainability = torch.squeeze(trainability)
            loss = (trainability+0.5) * loss
            loss = torch.sum(loss)

        trainable_value = {name:param for name, param in self.model.named_parameters() if 'mean' not in name}
        length = len(trainable_value.keys())
        trainable_value_patch = {}
        for name,param in self.patchnet.named_parameters():
            if name.split('.')[0]=='body':
                if int(name.split('.')[3])%3==0:
                    
                    trainable_value_patch[name]=param
            else:
                trainable_value_patch[name]=param
        
            
        if not args.patchnet1:
            trainable_value.update(trainable_value_patch)
            grad = torch.autograd.grad(loss, trainable_value.values()) #build full graph support gradient of gradient
            
            trainable_value_patch = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad[length:],list(trainable_value.values())[length:], list(trainable_value.keys())[length:])))
            train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad[0:length],list(trainable_value.values())[0:length], list(trainable_value.keys())[0:length])))
        else:
           
            grad = torch.autograd.grad(loss, trainable_value.values())
            train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad,list(trainable_value.values()), list(trainable_value.keys()))))
 
        for task_step in range(1,2):
            last_update = 1
            sr = self.model(lr,0,num,train_var)
            loss = self.loss(sr, hr) 
            if self.args.patchnet:
                trainability = self.patchnet(sr,trainable_value_patch)
                trainability = torch.squeeze(trainability)
                loss = (trainability+0.5) * loss
                loss = torch.sum(loss)

            if args.patchnet1 and task_step == last_update:
                grad = torch.autograd.grad(loss, train_var.values())
                train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad[0:length],list(train_var.values())[0:length], list(train_var.keys())[0:length])))
            else:
            
                train_var.update(trainable_value_patch)
                grad = torch.autograd.grad(loss, train_var.values())
                trainable_value_patch = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad[length:],list(train_var.values())[length:], list(train_var.keys())[length:])))
                train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad[0:length],list(train_var.values())[0:length], list(train_var.keys())[0:length])))
               
        
        
        sr = self.model(lr,0,num, train_var)
        
        
        return sr

    
    def set_forward_loss_attention(self, lr, hr, num, epoch):
        sr = self.set_forward_attention(lr, hr, num, epoch, is_feature = False)
        loss = self.loss(sr, hr)

        return loss

    
    def set_forward(self, lr , hr, num, epoch, is_feature = False):
       
        if int(epoch) == int(self.args.decay.split('-')[0]):
            self.args.inner_lr = self.args.inner_lr_decay

       
        sr = self.model(lr,0,num)
        loss = self.loss(sr,hr)

        trainable_value = {name:param for name, param in self.model.named_parameters() if 'mean' not in name}
       
        grad = torch.autograd.grad(loss, trainable_value.values())
       
        train_var = dict(map(lambda p: (p[2], p[1] - self.args.inner_lr * p[0]), zip(grad,trainable_value.values(), trainable_value.keys())))
        for task_step in range(1,2):
            sr = self.model(lr,0,num,train_var)
            loss = self.loss(sr, hr) 

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
       
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, num,) in enumerate(self.loader_train):
           
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

           
           
            if self.args.istype4few50:
                num1 = [i for i in range(len(num)) if 1<=int(num[i])<=50]
                num2 = [i for i in range(len(num)) if 51<=int(num[i])<=100]
                num3 = [i for i in range(len(num)) if 101<=int(num[i])<=150]
                num4 = [i for i in range(len(num)) if 151<=int(num[i])<=200]
                num5 = [i for i in range(len(num)) if 201<=int(num[i])<=250]
                num6 = [i for i in range(len(num)) if 251<=int(num[i])<=300]
                num7 = [i for i in range(len(num)) if 301<=int(num[i])<=350]
                num8 = [i for i in range(len(num)) if 351<=int(num[i])<=400]
                num9 = [i for i in range(len(num)) if 401<=int(num[i])<=450]
                num10 = [i for i in range(len(num)) if 451<=int(num[i])<=500]
                num11 = [i for i in range(len(num)) if 501<=int(num[i])<=550]
                num12 = [i for i in range(len(num)) if 551<=int(num[i])<=600]
                num13 = [i for i in range(len(num)) if 601<=int(num[i])<=650]
                num14 = [i for i in range(len(num)) if 651<=int(num[i])<=700]
                num15 = [i for i in range(len(num)) if 701<=int(num[i])<=750]
                num16 = [i for i in range(len(num)) if 751<=int(num[i])<=800]
                num17 = [i for i in range(len(num)) if 801<=int(num[i])<=850]
                num18 = [i for i in range(len(num)) if 851<=int(num[i])<=900]
                num19 = [i for i in range(len(num)) if 901<=int(num[i])<=950]
                num20 = [i for i in range(len(num)) if 951<=int(num[i])<=1000]
           
            
        
            if self.args.maml and self.args.use_maml:
                
                if self.args.istype4few50:
                    numlist = [(num1,0),(num2,1),(num3,2),(num4,3),(num5,4),(num6,5),(num7,6),(num8,7),(num9,8),(num10,9),(num11,10),(num12,11),(num13,12),(num14,13),(num15,14),(num16,15),(num17,16),(num18,17),(num19,18),(num20,19)]
               
                loss = 0
                
            
            
            if self.args.use_maml:
                if self.args.patchnet:
                    loss_all = []
                    avg_loss = 0
                    count = 0
                    for i in numlist:
                        if len(i[0])!=0:
                            loss = self.set_forward_loss_attention(lr[i[0]], hr[i[0]], i[1], epoch)
                            
                            avg_loss = avg_loss+loss.data
                            loss_all.append(loss)
                            count += 1
                    avg_loss /= count
                else:
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
                grad_mask(self.model,self.args)


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
                    else:
                        flag = 0
                  
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
                self.psnr.append(str(epoch)+": "+str(self.ckp.log[-1, idx_data, idx_scale]))
                self.psnr.append(": ")
                self.psnr.append(self.ckp.log[-1, idx_data, idx_scale])

                file_name = self.args.dir_data.split('/')[-1]+'_group.txt'
                
                txt_dir = os.path.join('..',file_name)
                
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
            if self.args.save_every:
                self.ckp.save_everyepoch(self, epoch, is_best=True)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    

    def test_pn(self):
        torch.set_grad_enabled(False)

        self.model.eval()
        

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
                    else:
                        flag = 0
                   
                    sr = self.model(lr, idx_scale, flag)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    I_frames = {'black':[1,79,264,382,485,604,671,830,920,1170,1213,1310],'lol_45s':[1,250,500,750,1000,1250],'izone':[1,179,309,472,518,571,663,688,852,953,1015,1041,1082,1108,1147,1218,],'emma_45s':[1,250,500,691,941,1168],'sport':[1,114,220,288,508,565,815,906,949,976,1028,1076,1317],'lon':[1,250,500,750,1000,1250],'lol_2min':[1,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500],'emma_30min':['00001', '00150', '00287', '00437', '00587', '00687', '00737', '00887', '01048', '01198', '01348', '01417', '01498', '01584', '01648', '01798', '01948', '02098', '02212', '02248', '02513', '02554', '02677', '02827', '02977', '03058', '03127', '03277', '03441', '03591', '03642', '03741', '03813', '03891', '04018', '04168', '04268', '04317', '04405', '04431', '04467', '04617', '04653', '04767', '04795', '04917', '05052', '05202', '05336', '05490', '05592', '05640', '05790', '05940', '05987', '06089', '06239', '06277', '06389', '06539', '06574', '06689', '06839', '06908', '06977', '07117', '07225', '07267', '07417', '07567', '07717', '07867', '07950', '08009', '08159', '08318', '08468', '08622', '08772', '08891', '08922', '09072', '09222', '09372', '09522', '09615', '09672', '09822', '09986', '10136', '10283', '10347', '10422', '10572', '10722', '10763', '10872', '10923', '11024', '11184', '11334', '11373', '11484', '11634', '11784', '11934', '12084', '12234', '12384', '12534', '12684', '12834', '12984', '13134', '13284', '13422', '13544', '13694', '13856', '14007', '14033', '14157', '14298', '14448', '14598', '14748', '14898', '15048', '15198', '15362', '15512', '15662', '15812', '15966', '16116', '16152', '16224', '16266', '16401', '16450', '16551', '16692', '16739', '16842', '16992', '17142', '17292', '17442', '17591', '17741', '17822', '17891', '18031', '18041', '18168', '18301', '18351', '18451', '18568', '18601', '18751', '18901', '19051', '19154', '19201', '19351', '19483', '19622', '19768', '19889', '20039', '20110', '20189', '20226', '20259', '20339', '20489', '20619', '20769', '20875', '20889', '21039', '21189', '21280', '21339', '21441', '21489', '21639', '21710', '21770', '21843', '21924', '22074', '22220', '22251', '22299', '22370', '22410', '22522', '22672', '22822', '22936', '22972', '23046', '23122', '23275', '23425', '23575', '23725', '23875', '24025', '24175', '24325', '24488', '24638', '24788', '24916', '25066', '25230', '25357', '25421', '25507', '25559', '25657', '25730', '25807', '25849', '25957', '26107', '26257', '26336', '26407', '26556', '26706', '26856', '27006', '27156', '27280', '27380', '27430', '27580', '27730', '27880', '27985', '28030', '28180', '28268', '28330', '28480', '28578', '28630', '28706', '28766', '28804', '28916', '28963', '29066', '29109', '29216', '29328', '29369', '29519', '29669', '29717', '29812', '29962', '30112', '30262', '30412', '30439', '30554', '30563', '30727', '30786', '30884', '31034', '31157', '31250', '31301', '31451', '31601', '31668', '31720', '31751', '31817', '31872', '32022', '32111', '32172', '32322', '32472', '32581', '32622', '32709', '32772', '32875', '32922', '32972', '33075', '33180', '33204', '33255', '33354', '33477', '33605', '33755', '33905', '33970', '34055', '34151', '34205', '34258', '34355', '34395', '34505', '34630', '34765', '34791', '34866', '34941', '35091', '35196', '35241', '35391', '35547', '35589', '35697', '35847', '35968', '35994', '36157', '36282', '36383', '36430', '36560', '36619', '36710', '36860', '37010', '37044', '37114', '37160', '37312', '37462', '37533', '37612', '37762', '37912', '38070', '38220', '38299', '38370', '38520', '38632', '38670', '38828', '38936', '38978', '39020', '39128', '39278', '39428', '39578', '39620', '39728', '39777', '39878', '39971', '40028', '40178', '40214', '40246', '40328', '40415', '40478', '40628', '40698', '40775', '40838', '40918', '40944', '41068', '41218', '41368', '41518', '41668', '41758', '41818', '41968', '42118', '42268', '42332', '42418', '42539', '42689', '42839', '42888', '42989', '43139', '43289', '43353', '43436', '43586', '43736', '43886', '44011', '44024', '44174', '44324', '44389', '44456', '44606', '44756', '44906', '45056', '45180', '45206', '45356', '45476', '45626', '45746', '45896', '46055', '46136', '46205', '46235', '46286', '46355', '46505', '46655', '46805', '46955', '47092', '47186', '47226', '47376', '47526', '47676', '47788', '47825', '47945', '47955', '48060', '48105', '48149', '48255', '48387', '48537', '48641', '48687', '48837', '48956', '48987', '49137', '49287', '49409', '49559', '49584', '49709', '49847', '49995', '50145', '50295', '50445', '50595', '50745', '50885', '51035', '51185', '51292', '51335', '51485', '51635', '51785', '51874', '51935', '52085', '52160', '52208', '52358', '52359']}
                    keys = I_frames.keys()
                    for key in keys:
                        if key in self.args.dir_data:
                            I_frame = I_frames[key]
                            continue
                    
                    if int(filename[0]) in I_frame:
                       
                        sr = sr.squeeze().permute(2,1,0)
                        hr = hr.squeeze().permute(2,1,0)
                        hr_patch_size = 144
                        shave = scale + 6
                        patch_size = hr_patch_size - shave*2
                        diff_norm = (sr - hr) / self.args.rgb_range

                        diff_norm = diff_norm[shave:-shave, shave:-shave, ...]

                    

                        dpm = utility.cal_dp_map(np.array(diff_norm.cpu()), patch_size)
                       
                        sum_map = utility.box_filter(dpm,patch_size)

                        eps = 1e-9
                        psnr_map = utility.cal_psnr_map(sum_map,scale,eps)
                        [hei, wid] = psnr_map.shape

                        iy = np.arange(hei).reshape(-1,1).repeat(wid,axis=1).reshape(-1,1)
                        ix = np.arange(wid).reshape(1,-1).repeat(hei,axis=0).reshape(-1,1)

                        index_psnr = utility.psnr_sort(psnr_map, iy, ix)
                        print(int(filename[0]))
                    psnr_path = os.path.join(self.args.dir_data,"psnr_map")
                    if not os.path.exists(psnr_path):
                        os.mkdir(psnr_path)
                    psnr_file = os.path.join(self.args.dir_data,"psnr_map",filename[0]+"_psnr_map.pt")

                    with open(psnr_file, 'wb') as _f:
                        pickle.dump(index_psnr, _f)

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
