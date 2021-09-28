import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import numpy as np
import scipy.stats as st



def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not

    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0
    #print(type(K1),type(data_range))
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)
    mu1 = gaussian_filter(X, win)
#    print('iamhere')
    mu2 = gaussian_filter(Y, win)
#    print('iamthere')
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not

    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val

class SSIM_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image):
               
        loss = 1 - ssim(out_image,gt_image)
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class Regionloss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_true, y_pred):
        w = y_true.shape[2]
        h = y_true.shape[3]
        percent = 0.4
        index = int(w * h * percent - 1)
        gray1 = (y_pred[:,0, :, :] + y_pred[:,1, :, :] + y_pred[:,2, :, :])/3
        gray = torch.reshape(gray1, [-1, w * h])
        
        gray_sort = torch.topk(-gray, w * h)[0]
        yu = gray_sort[:, index]
        yu = yu.unsqueeze_(-1).unsqueeze_(-1)
        mask = gray1 <= -yu
        mask1 = mask.unsqueeze_(1)
        mask = torch.cat([mask1, mask1, mask1], 1).float().cuda()
        low_fake_clean = mask*y_pred
        high_fake_clean = (1-mask)*y_pred
        low_clean = mask*y_true
        high_clean = (1 - mask)*y_true
        Region_loss = torch.mean(torch.abs(low_fake_clean - low_clean) * 0.8 +0.2* torch.abs(high_fake_clean - high_clean))
        
        return Region_loss
    
class TLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x,y):
        x = 0.114*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.299*x[:,2,:,:]
        y = 0.114*y[:,0,:,:] + 0.587*y[:,1,:,:] + 0.299*y[:,2,:,:]
        batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = self._tensor_size(x[:,1:,:])
        count_w = self._tensor_size(x[:,:,1:])
        hx_tv = x[:,1:,:]-x[:,:h_x-1,:]
        wx_tv = x[:,:,1:]-x[:,:,:w_x-1]
        hy_tv = y[:,1:,:]-y[:,:h_x-1,:]
        wy_tv = y[:,:,1:]-y[:,:,:w_x-1]
        h_tv  = torch.pow((wx_tv-wy_tv),2).sum()
        w_tv  = torch.pow((wx_tv-wy_tv),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]