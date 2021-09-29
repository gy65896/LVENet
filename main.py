# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:34:04 2020

@author: Administrator
"""

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
from utils import *
from loss import *

#from torchstat import stat

#调用GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#用于加载模型
def load_checkpoint(checkpoint_dir, learnrate):
    if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
        
        #加载存在的模型
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
        net = IEM()
        
        #stat(net, (3, 224, 224))

        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']
            
    else:
        # 创建模型
        net = IEM()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
        cur_epoch = 0
    return model, optimizer,cur_epoch

def test(argspar, model, epoch = False):
    if argspar.syn:
        files_clear = os.listdir(argspar.input_tests)     
        for i in range(len(files_clear)):   
            model.eval()
            with torch.no_grad():
                name = files_clear[i][:-4]
                clear = cv2.imread(argspar.input_tests + '/' + files_clear[i])
                low, clear, L_real = syn_low(clear, 'test',0.4,0.4)
                #模型处理
                starttime = time.clock()
                R_out, L_out = model(low) 
                endtime1 = time.clock()
                
                print('The ' + name+' Time:' +str(endtime1-starttime)+'s.'+\
                      'PSNR:%.4f'%(tensor_psnr(clear, R_out)))
                temp = R_out#torch.cat((clear, low, L_real, L_out, R_out), dim = 3)
                temp = tensor2img(temp)
                if epoch:
                    cv2.imwrite(argspar.output_test + '/' + files_clear[i][:-4] +'_%d'%(epoch+1)+files_clear[i][-4:],\
                                np.clip(temp*255,0.0,255.0))
                else:
                    cv2.imwrite(argspar.output_test + '/' + files_clear[i][:-4] +'_LVENet'+files_clear[i][-4:],\
                                np.clip(temp*255,0.0,255.0))
    else:
        files_low = os.listdir(argspar.input_testr)
        for i in range(len(files_low)):   
            model.eval()
            with torch.no_grad():
                name = files_low[i][:-4]
                low = cv2.imread(argspar.input_testr + '/' + files_low[i])
                low = img2tensor(low)
                
                starttime = time.clock()
                R_out, L_out = model(low) 
                R_out = torch.clamp(low*(torch.clamp(L_out,0.15,1)**(-1)),0,1)
                endtime1 = time.clock()
                Result2 = R_out*(1-L_out)+L_out*low
                
                print('The ' + name+' Time:' +str(endtime1-starttime)+'s.')
                temp = Result2
                temp = tensor2img(temp)
                if epoch:
                
                    cv2.imwrite(argspar.output_test + '/' + files_low[i][:-4] +'_%d'%(epoch+1)+files_low[i][-4:],\
                                np.clip(temp*255,0.0,255.0))
                else:
                    cv2.imwrite(argspar.output_test + '/' + files_low[i][:-4] +'_LVENet'+files_low[i][-4:],\
                                np.clip(temp*255,0.0,255.0))

def train(model, optimizer, cur_epoch, argspar, loader_train_syn):
    #损失函数定义
    ssim_loss = SSIM_loss().cuda()
    Region_loss=Regionloss().cuda()
    T_loss = TLoss().cuda()
    
    #开始训练
    for epoch in range(cur_epoch, argspar.epoch):#批次数
        optimizer = adjust_learning_rate(optimizer, epoch, argspar.lr)
        learnrate = optimizer.param_groups[-1]['lr']
        model.train()
        #合成图像训练
        for i,data in enumerate(loader_train_syn,0):
            #合成图像
            low, clear, L_real = syn_low(data,'train',0.4,0.8)
            #模型输出
            R_out, L_out = model(low)
            #计算损失函数
            loss = 2*ssim_loss(R_out, clear) + 0.5*T_loss(L_out, L_real)\
                + 1.3*Region_loss(L_out, L_real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[epoch %d][%d/%d] lr :%f loss: %.4f PSNR: %.6f"%(epoch+1, i+1, \
                len(loader_train_syn), learnrate, loss.item(),tensor_psnr(R_out, clear)))
        
        #保存模型
        save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}, argspar.model, is_best=0)
        if (epoch + 1) % 1 == 0:
            test(argspar, model, epoch)
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "crop image")
    parser.add_argument("--train", type=str, default =  False, help = '训练还是测试')
    
    parser.add_argument("--epoch", type=int, default = 60, help = '总批次数')
    parser.add_argument("--batch_size", type=str, default =1, help = '批次大小')
    parser.add_argument("--learn_rate", type=str, default = 1e-3, help = '学习率')
    parser.add_argument("--lr", type=int, default = 30, help = '调整学习率批次数')
    
    parser.add_argument("--model", type=str, default = "./checkpoint/", help = '模型1保存路径')

    
    parser.add_argument("--syn", type=str, default = True, help = '合成还是真实低照度')
    
    parser.add_argument("--input_tests", type=str, default = "./data/input/syn", help = '测试输入路径')
    parser.add_argument("--input_testr", type=str, default = "./data/input/real", help = '测试输入路径')
    parser.add_argument("--output_test", type=str, default = "./Result", help = '测试输出路径')
    argspar = parser.parse_args()
    
    print("\n低照度增强网络")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    argspar = parser.parse_args()
    if argspar.train:
        #开始训练
        print('> Loading dataset ...')
        #加载训练参数
        dataset_train_syn = Dataset(trainsyn=True, shuffle=True)
        #放置模型和设置批次数
        loader_train_syn = DataLoader(dataset=dataset_train_syn, num_workers=0,\
                                  batch_size=argspar.batch_size, shuffle=True)
        model, optimizer, cur_epoch = load_checkpoint(argspar.model, \
                                        argspar.learn_rate)
        #训练IEM
        start = time.clock()
        train(model, optimizer, cur_epoch, argspar, loader_train_syn)
        end = time.clock()

        print('The whloe Time:' +str(end-start)+'s.')
    else:
        model, optimizer, cur_epoch = load_checkpoint(argspar.model, \
                                        argspar.learn_rate)
        test(argspar, model)