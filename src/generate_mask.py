import torch
import numpy as np
import time
from option import args
import os
p_mask = args.p_mask

def find_gradmask(m1,m2):
    
    dist_mean = 0
    count = 0
    dist_w = {}
    temp=None
    start = time.time()
    for key in m1.keys():
        if 'weight' in key and 'PReLU' not in key:
            count += 1
            a = m1[key]
            b = m2[key]
            dist = torch.abs(a-b)
            
            dist_w[key] = dist
            dist = dist.view(1,-1)
            if temp!=None:
                dist = torch.cat((dist,temp),1)
            temp = dist
            
            
    
    dist_mean,_ = dist.sort(descending=True) #sort mean of each conv
    idx = int(len(dist_mean[0]) * p_mask)    
    pivot = dist_mean[0][idx]
    print('pivot',pivot)
    pctg = idx/len(dist_mean[0])*100
    print('percentage',pctg)


    mask = {}
    for key in m1.keys():
        if 'weight' in key and 'PReLU' not in key:
            w = dist_w[key]
            mask_w = m1[key]
            a,b,c,d = w.shape
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        for m in range(d):
                            if w[i][j][k][m] < pivot:
                                mask_w[i][j][k][m] = 0
                            else:
                                mask_w[i][j][k][m] = 1
            key = key.replace('model.','')
            mask[key] = mask_w
        if 'bias' in key and 'bias.weight' not in key:
            mask_b = m1[key]
            mask_b = 0
            key = key.replace('model.','')
            mask[key] = mask_b

    save_name = args.dir_mask
    file = open(save_name,'a+')
    torch.save(mask, save_name)
    end = time.time()
    print((end-start))
    print('grad mask save successful!')

                    
