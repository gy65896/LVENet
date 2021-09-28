import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import cv2
import h5py
import scipy.misc
from model import *
#from data.train.makedataset import Dataset
import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.measure.simple_metrics import compare_psnr
from copy import copy
import torch.nn.functional as F

def tensor2img(img):
    Out = img.squeeze().cpu().detach().numpy()
    Out = np.transpose(Out, axes=[1, 2, 0])
    return Out

def img2tensor(img):
    img = np.array(img/255.)
    Out = np.transpose(img, axes=[2, 0, 1])
    Out = torch.from_numpy(Out.copy()).type(torch.FloatTensor).unsqueeze(0)
    with torch.no_grad():
        Out = Variable(Out.cuda(),requires_grad=True)
    return Out

def save_checkpoint(state, checkpoint, is_best, filename='checkpoint.pth.tar'):#保存学习率
    torch.save(state, checkpoint + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(checkpoint + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):#调整学习率
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            print( param_group['lr'])
    return optimizer

def tensor_psnr(img, imclean, data_range=1):#计算图像PSNR输入为Tensor
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
        minimum and maximum possible values). By default, this is estimated
        from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],data_range=data_range)
    return psnr/img_cpu.shape[0]

def Illumination(L,method):
    if method == "max_c":
        return np.max(L,axis=2)
    elif method == "min_c":
        return np.min(L,axis=2)
    else:
        print("输入模式有误请输入max_c or min_c")

def guideFilter(I, p, winSize, eps):

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b
    return q

def syn_low_I(img,min,max):
    img   = np.transpose(img, (1, 2, 0))	
    I_H   = Illumination(img,"max_c")
    img_g = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    I_H   = guideFilter(I_H, img_g, (7,7), 100)
    I_H   = I_H[:,:,np.newaxis]
    img_max = np.max(I_H)
    img_min = np.min(I_H)
    I_H = np.float32((I_H - img_min) / np.maximum((img_max - img_min), 0.001))
    Low   = np.random.uniform(min,max)
    I_L   = np.maximum(I_H*Low,0.05)
    #noise=np.random.normal(sigmin,sigmax, img.shape)
    #img_noise = img+noise
    out   = img*I_L
    return np.transpose(out, (2, 0, 1)),np.transpose(I_L, (2, 0, 1))	

def syn_low(clear, model, min, max):
    #图像导入
    if model == 'train':
        clear = clear.numpy()
        Low = np.zeros(clear.shape)
        I = np.zeros(clear.shape)
        for nx in range(clear.shape[0]):
            #合成图像
            Low[nx,:,:,:],I[nx,:,:,:]=syn_low_I(clear[nx,:,:,:],min,max)
        Low = torch.from_numpy(Low.copy()).type(torch.FloatTensor)
        clear = torch.from_numpy(clear.copy()).type(torch.FloatTensor)
        I = torch.from_numpy(I.copy()).type(torch.FloatTensor)
        with torch.no_grad():
            Low = Variable(Low.cuda(),requires_grad=True)
            clear = Variable(clear.cuda(),requires_grad=True)
            I = Variable(I.cuda(),requires_grad=True)
        return Low, clear,I #输出雾气、透射率、大气光值、清晰图像
    else:
        clear = (np.array(clear, dtype="float32") / 255.0).transpose(2, 0, 1)	
        Low   = np.zeros(clear.shape)
        Low,I = syn_low_I(clear,min,max)
        Low   = Low[np.newaxis,:,:,:]
        clear   = clear[np.newaxis,:,:,:]
        Low = torch.from_numpy(Low.copy()).type(torch.FloatTensor)
        clear = torch.from_numpy(clear.copy()).type(torch.FloatTensor)
        I = torch.from_numpy(I.copy()).type(torch.FloatTensor)
        with torch.no_grad():
            Low = Variable(Low.cuda(),requires_grad=True)
            clear = Variable(clear.cuda(),requires_grad=True)
            I = Variable(I.cuda(),requires_grad=True)
        return Low, clear,torch.cat((I[:,np.newaxis,:,:],I[:,np.newaxis,:,:],I[:,np.newaxis,:,:]),1) #输出雾气、透射率、大气光值、清晰图像
        