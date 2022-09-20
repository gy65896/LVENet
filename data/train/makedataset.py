# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:00:46 2020

@author: Administrator
"""

import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata

class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()

		self.trainsyn = trainsyn
		self.train_syn_rgb	 = './data/train/train_syn.h5'
		self.train_real_rgb	 = './data/train/train_real.h5'
		if self.trainsyn:
			h5f = h5py.File(self.train_syn_rgb, 'r')
		else:
			h5f = h5py.File(self.train_real_rgb, 'r')				 		  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		if self.trainsyn:
			h5f = h5py.File(self.train_syn_rgb, 'r')
		else:
			h5f = h5py.File(self.train_real_rgb, 'r')		  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)


def data_augmentation(image, mode):
	r"""Performs dat augmentation of the input image

	Args:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
			0 - no transformation
			1 - flip up and down
			2 - rotate counterwise 90 degree
			3 - rotate 90 degree and flip up and down
			4 - rotate 180 degree
			5 - rotate 180 degree and flip
			6 - rotate 270 degree
			7 - rotate 270 degree and flip
	"""
	out = np.transpose(image, (1, 2, 0))
	if mode == 0:
		# original
		out = out
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1))

def data_augmentation_real(image, low, mode):
	r"""Performs dat augmentation of the input image

	Args:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
			0 - no transformation
			1 - flip up and down
			2 - rotate counterwise 90 degree
			3 - rotate 90 degree and flip up and down
			4 - rotate 180 degree
			5 - rotate 180 degree and flip
			6 - rotate 270 degree
			7 - rotate 270 degree and flip
	"""
	out = np.transpose(image, (1, 2, 0))
	oul = np.transpose(low,   (1, 2, 0))
	if mode == 0:
		# original
		out = out
		oul = oul
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
		oul = np.flipud(oul)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
		oul = np.flipud(oul)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
		oul = np.rot90(oul)
		oul = np.flipud(oul)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
		oul = np.rot90(oul, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
		oul = np.rot90(oul, k=2)
		oul = np.flipud(oul)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
		oul = np.rot90(oul, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
		oul = np.rot90(oul, k=3)
		oul = np.flipud(oul)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1)), np.transpose(oul, (2, 0, 1))#np.transpose(out, (2, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
    
    chl,raw,col = img.shape
    num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
    num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
    count = 0
    total_process = int(num_col)*int(num_raw)
    img_patches = np.zeros([chl,win,win,total_process])
    if Syn:
        for i in range(num_raw):
            for j in range(num_col):               
                if stride * i + win <= raw and stride * j + win <=col:
                    img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]
                elif stride * i + win > raw and stride * j + win<=col:
                    img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]
                elif stride * i + win <= raw and stride*j + win>col:
                    img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
                else:
                    img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]
                count +=1     
                
    return img_patches

def img_to_patches_real(img,low,win,stride,Syn=True):
    
    chl,raw,col = img.shape
    num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
    num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
    count = 0
    total_process = int(num_col)*int(num_raw)
    img_patches = np.zeros([chl,win,win,total_process])
    low_patches = np.zeros([chl,win,win,total_process])
    if Syn:
        for i in range(num_raw):
            for j in range(num_col):               
                if stride * i + win <= raw and stride * j + win <=col:
                    img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]
                    low_patches[:,:,:,count] = low[:,stride*i : stride*i + win, stride*j : stride*j + win]
                elif stride * i + win > raw and stride * j + win<=col:
                    img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]
                    low_patches[:,:,:,count] = low[:,raw-win : raw,stride * j : stride * j + win]
                elif stride * i + win <= raw and stride*j + win>col:
                    img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
                    low_patches[:,:,:,count] = low[:,stride*i : stride*i + win, col-win : col]
                else:
                    img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]
                    low_patches[:,:,:,count] = low[:,raw-win : raw,col-win : col]
                count +=1     
                
    return img_patches, low_patches

def readfiles(filepath):
	'''Get dataset images names'''
	files = os.listdir(filepath)
	return files

def normalize(data):
	
	return np.float32(data/255.)

def samesize(img,size):
    
    img = cv2.resize(img,size)
    return img

def concatenate2imgs(cimg,nimg):
	
	c,w,h = cimg.shape
	conimg = np.zeros((2*c,w,h))
	conimg[0:c,:,:] = cimg
	conimg[c:2*c,:,:] = nimg
	
	return conimg

def TrainSynRGB(filepath_clear,patch_size,stride):
	'''synthetic RGB images'''
	train = 'train_syn.h5'
	files_clear = readfiles(filepath_clear)
	count = 0
	scales = [0.8,1.0]	  
		
	with h5py.File(train, 'w') as h5f:
		for i in range(len(files_clear)):
			clear = cv2.imread(filepath_clear + '/' + files_clear[i])
			clear = samesize(clear,(360,360))
			for sca in scales:
				img_clear = cv2.resize(clear, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)	
				#img_clear = (cv2.cvtColor(img_clear, cv2.COLOR_BGR2RGB))
				img_clear = normalize(img_clear)

				img_patches = img_to_patches(img_clear, win=patch_size, stride=stride)
				print("\tfile: %s scale %.1f # samples: %d" %(files_clear[i], sca,img_patches.shape[3]))	  
				for nx in range(img_patches.shape[3]):
					data = data_augmentation(img_patches[:, :, :, nx].copy(), np.random.randint(0, 7))
					#cv2.imwrite('./ttt/1.png',np.clip(data[:,:,::-1]*255,0,255))
					print(data.shape)
					h5f.create_dataset(str(count), data=data)
					count += 1
			i += 1
		print('\n> Total')
		print('\ttraining set, # samples %d' % count)
	h5f.close()

def TrainRealRGB( filepath_clear, filepath_low,patch_size,stride):
	'''synthetic RGB images'''
	train = 'train_real.h5'
	files_clear = readfiles(filepath_clear)
	files_low = readfiles(filepath_low)
	count = 0
	scales = [0.8,1.0]	  
		
	with h5py.File(train, 'w') as h5f:
		for i in range(len(files_clear)):
			clear = cv2.imread(filepath_clear + '/' + files_clear[i])
			clear = samesize(clear,(360,360))
			low = cv2.imread(filepath_low + '/' + files_clear[i])
			low = samesize(low,(360,360))
			for sca in scales:
				img_clear = cv2.resize(clear, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)	
				#img_clear = (cv2.cvtColor(img_clear, cv2.COLOR_BGR2RGB))
				img_clear = normalize(img_clear)
                
				img_low   = cv2.resize(low, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)	
				#img_low   = (cv2.cvtColor(img_low, cv2.COLOR_BGR2RGB))
				img_low   = normalize(img_low)
				#img_depth = normalize(img_depth)
				img_patches, low_patches = img_to_patches_real(img_clear, img_low, win=patch_size, stride=stride)
				print("\tfile: %s scale %.1f # samples: %d" %(files_clear[i], sca,img_patches.shape[3]))	  
				for nx in range(img_patches.shape[3]):
					data1,data2 = data_augmentation_real(img_patches[:, :, :, nx].copy(), low_patches[:, :, :, nx].copy(), np.random.randint(0, 7))
					data = np.concatenate((data1, data2), axis=0)
					#print(data.shape)
					#cv2.imwrite('./ttt/1.png',np.clip(data1*255,0,255))
					#cv2.imwrite('./ttt/2.png',np.clip(data2[:,:,::-1]*255,0,255))
					h5f.create_dataset(str(count), data=data)
					count += 1
			i += 1
		print('\n> Total')
		print('\ttraining set, # samples %d' % count)
	h5f.close()