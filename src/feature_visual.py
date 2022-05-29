import numpy as np
from numpy.linalg import norm
import torch
from option import args
import model
import cv2 as cv

def cos_dis(f1, f2):
    features1 = f1.numpy().reshape(f1.shape[0],-1)
    features2 = f2.numpy().reshape(f1.shape[0],-1)
    norm1 = norm(features1,axis=-1).reshape(features1.shape[0],1)
    norm2 = norm(features2,axis=-1).reshape(1,features2.shape[0])
    end_norm = np.dot(norm1,norm2)
    cos = np.dot(features1, features2.T)/end_norm
    return cos

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    checkpoint = utility.checkpoint(args)
    model1 = model.Model(args, checkpoint)
    args.pre_train = ""
    checkpoint = utility.checkpoint(args)
    model2 = model.Model(args, checkpoint)    
    model1.eval()
    model2.eval()
    img = cv.imread("")
    
