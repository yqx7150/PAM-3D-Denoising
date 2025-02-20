from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import cv2

from matplotlib.image import imread

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import math
import scipy.io as io


# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  ckpt_filename = "exp(old)/checkpoints/checkpoint_1.pth"
  
  #ckpt_filename = "./exp(old)/checkpoints-sigmax378/checkpoint_2.pth"
  print(ckpt_filename)
  if not os.path.exists(ckpt_filename):
      print('!!!!!!!!!!!!!!'+ckpt_filename + ' not exists')
      assert False
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
  
  
batch_size = 1 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())


#@title PC inpainting

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.21#0.07#0.075 #0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

pc_inpainter = controllable_generation.get_pc_inpainter(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True)  


os.makedirs('./results/inpainter/', exist_ok=True)
os.makedirs('./results/Rec/', exist_ok=True)
os.makedirs('./results/hankle/', exist_ok=True)

def save_img(img, img_path):

    img = np.clip(img*255,0,255)    ##最小值0, 最大值255

    cv2.imwrite(img_path, img)

#####make_dsr_rgb#####
#作用是得到pading后的dimg和vid.mid datap 找到图片数据位置，定位。
# 这个函数在这里估计没有用不到  
# def make_dsr_rgb(dimg,Nimg,Nfir,data):

#     Nshrink=Nimg-Nfir+1;  # 25
#     if Nimg%2==0:
#         hNimg=int(Nimg/2)
#     else:
#         hNimg=int((Nimg-1)/2)  # 18

    
    
        
#     Ny,Nx,Nc=dimg.shape
    
    
#     arr0 = np.pad(dimg[:,:,0],((hNimg,hNimg),(hNimg,hNimg)),'constant')
#     arr1 = np.pad(dimg[:,:,1],((hNimg,hNimg),(hNimg,hNimg)),'constant')
#     arr2 = np.pad(dimg[:,:,2],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    
#     dimgp = np.concatenate((arr0[:,:,np.newaxis],arr1[:,:,np.newaxis],arr2[:,:,np.newaxis]),2)
    
#     arr3 = np.pad(data[:,:,0],((hNimg,hNimg),(hNimg,hNimg)),'constant')
#     arr4 = np.pad(data[:,:,1],((hNimg,hNimg),(hNimg,hNimg)),'constant')
#     arr5 = np.pad(data[:,:,2],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    
#     datap = np.concatenate((arr3[:,:,np.newaxis],arr4[:,:,np.newaxis],arr5[:,:,np.newaxis]),2)
    
    
    
    
#     # plt.imshow(dimgp)
#     # plt.show()
    
#     #vmask_arr = np.ones((256,256))
#     vmask_arr = np.ones((512,512))
#     vmask_arr1 = np.pad(vmask_arr[:,:],((hNimg,hNimg),(hNimg,hNimg)),'constant')    
#     vmask = np.concatenate((vmask_arr1[:,:,np.newaxis],vmask_arr1[:,:,np.newaxis],vmask_arr1[:,:,np.newaxis]),2)  
    
    
#     M1 = vmask
#     M1 = np.reshape(M1.T,[M1.shape[0]*M1.shape[1]*M1.shape[2],1])
#     vid = np.where(M1 == 1 )[0]
    
#     mmask=np.zeros((dimg.shape[0],dimg.shape[1]))
    
#     eNshrink=math.floor(Nshrink/2)
#     hNshrink=math.floor(Nshrink/1)
    
    
#     # 这里注意NX NY用的一样的
#     if int(Nshrink/2)==0:
#         # mmask(np.arange(eNshrink,Ny-eNshrink,hNshrink),np.arange(eNshrink,Ny-eNshrink,hNshrink))=1;
#         # mmask(np.arange(eNshrink,Ny-eNshrink,hNshrink),Nx-eNshrink)=1;
#         # mmask(Ny-eNshrink,np.arange(eNshrink,Ny-eNshrink,hNshrink))=1;
#         #assert 0
#         mmask[Ny-eNshrink,Nx-eNshrink]=1;
#     else:
#         for iii in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
            
#             for jjj in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
#                 mmask[iii,jjj]=1
#         # 
#         for aaa in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
#             mmask[aaa,Nx-eNshrink-1]=1
#         for bbb in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
#             mmask[Nx-eNshrink-1,bbb]=1

#         mmask[Ny-eNshrink-1,Nx-eNshrink-1]=1;
    
#     # print(np.argwhere(mmask==1),np.argwhere(mmask==1).shape)#,eNshrink.dtype)
    
#     mmask = np.pad(mmask,((hNimg,hNimg),(hNimg,hNimg)),'constant') 
    
#     M2 = mmask
#     M2 = np.reshape(M2.T,[M2.shape[0]*M2.shape[1],1])
#     mid = np.where(M2 == 1 )[0] 
    

#     # print(mid,mid)
#     # assert 0
    
#     return dimgp,vid,mid,datap

###分块###
def get_cut_data(data,flag):
    i=0
    j=0
    if flag<=3:
        i=0
        j=flag
    elif (flag<=7)and(flag>=3):
        i=1
        j=flag-4
    elif (flag<=11)and(flag>=7):
        i=2
        j=flag-8
    else:
        i=3
        j=flag-12

    data_s=np.zeros([64,64,3])
    data_s = data[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :]

    # if index==1:
    #     data_s = np.zeros([64, 64, coil], dtype=np.complex64)
    #     data_s = data[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :]
    # if index==2:
    #     data_s = np.zeros([coil,64, 64])
    #     data_s = data[:,i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]






    return data_s


###块拼接###

def get_recover_data(data,flag,sample_data):
    i = 0
    j = 0
    if flag <= 3:
        i = 0
        j = flag
    elif (flag <= 7) and (flag >= 3):
        i = 1
        j = flag - 4
    elif (flag <= 11) and (flag >= 7):
        i = 2
        j = flag - 8
    else:
        i = 3
        j = flag - 12

    #不知道这里是不是改对的，原设置了dtype
    rec_data=np.zeros([3, 512, 512])
    rec_data=sample_data
    rec_data[:,i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]=data
    print(data.shape)
    print(i*64,(i+1)*64,j*64,(j+1)*64)
    print(i,j)

    return rec_data




def aloha_patch_rgb(k_w):

    N=64
    rec_datu=np.zeros_like(k_w)
    for iter_num in range(N):

        k_w=get_cut_data(data,iter_num)
        
        size_data = [Nimg,Nimg,3]
        ksize= [Nfir,Nfir]
    ###savve_img可有可无，可以保存方便显示，函数还没有加入到这个代码中，
        save_img(k_w,'inpain.png')

        
        k_w = torch.from_numpy(k_w).cuda()
    
        
    

        # write_Data("./results/result_zero.txt", str(iter_num)+'-'+aaa, {"psnr_zero":psnr_zero, "ssim_zero":ssim_zero})
        
        
        
        ####没有了mask,第二歌aaa路径看后面，应该不用改，hNfir应该不用，
        X = pc_inpainter(score_model, aaa, ckpt_filename, k_w, size_data, ksize, iter_num)
        
        
  
    
        '''
        X=admm_hankel(U,V,meas,meas_id,mu,muiter,Nimg,Nfir)
        plt.figure()
        plt.subplot(1,2,1)

        plt.imshow(k_w[:,:,::-1])
     
        rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] = rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:]+ \
                                                            X[hNfir:int(X.shape[0]-hNfir)+1, hNfir:int(X.shape[1]-hNfir)+1, :]
        plt.subplot(1,2,2)

        plt.imshow(rimg[:,:,::-1])
        plt.ion()

        plt.pause(0.5)
        plt.close()
        '''
        
        # rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] = rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:]+ \
        #                                                     X[hNfir:int(X.shape[0]-hNfir)+1, hNfir:int(X.shape[1]-hNfir)+1, :]
        
        # map_count[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] = map_count[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] + 1
        
        rec_datu=get_recover_data(X,iter_num ,rec_datu)

        
        #assert 0
    
    
    
    # print(maskp,maskp.shape,maskp.dtype)


####??????#####

    # id = np.argwhere(map_count==0)
    # for num_id in range(len(id)):
    #     map_count[id[num_id][0], id[num_id][1], id[num_id][2]]=1


    # rimg_n=np.divide(rimg, map_count)

#############


    #save_img(rimg,'./rimg.png')
    #io.savemat('./rimg.mat',{'data':rimg})
    # return rimg_n
    return rec_datu



####分块后进入pc，出来后拼接成一个整体。





#####开始运行###
psnr_all = []
ssim_all = []
mse_all=[]
nmse_all=[]

path='./lzdata/wgj_test'

Nimg=64 #37;
Nfir=8  #13; 
mu = 1e1
muiter = 50


## 图像填充，填充成512*512*3
def pad_image(dimg, dup, ddown, dleft, dright):
    arr0 = np.pad(dimg[:, :, 0], ((dup, ddown), (dleft, dright)), 'constant')
    arr1 = np.pad(dimg[:, :, 1], ((dup, ddown), (dleft, dright)), 'constant')
    arr2 = np.pad(dimg[:, :, 2], ((dup, ddown), (dleft, dright)), 'constant')

    imgp = np.concatenate((arr0[:, :, np.newaxis], arr1[:, :, np.newaxis], arr2[:, :, np.newaxis]), 2)

    return imgp

## 图像填充，caijian成375*500*3
def depad_img(imgp, dup, ddown, dleft, dright):
    dpimg0 = imgp[dup:(512 - ddown), dleft:(512 - dright), 0]
    dpimg1 = imgp[dup:(512 - ddown), dleft:(512 - dright), 1]
    dpimg2 = imgp[dup:(512 - ddown), dleft:(512 - dright), 2]

    dpimg = np.concatenate((dpimg0[:, :, np.newaxis], dpimg1[:, :, np.newaxis], dpimg2[:, :, np.newaxis]), 2)
    return dpimg

for aaa in sorted(os.listdir(path)):

    file_path = os.path.join(path, aaa)
       
    
    siat_input=cv2.imread(file_path)
    data = siat_input / 255.
    
    print("#################################  yu  ##############")
    print('data.shape',data.shape)
    
    nx,ny,nz = data.shape    ## nx=375, ny=500, nz=3
    if nx%2 == 0 :
        pup = int((512 - nx) / 2)  ## 68
        pdown = int((512 - nx) / 2)  ## 69
    else:
        pup = int((512 - nx) / 2)  ## 68
        pdown = int((512 - nx) / 2 + 1)  ## 69
    if ny%2 == 0 :
        pleft = int((512 - ny) / 2)  ## 6
        pright = int((512 - ny) / 2)  ## 6
    else:
        pleft = int((512 - ny) / 2)  ## 6
        pright = int((512 - ny) / 2 + 1)  ## 6

####数据集中的大小就是512*512大小，所以datapad就是输入的图片
    imgp = pad_image(data, pup, pdown, pleft, pright)    ## 512*512*3
    
    
    datapad = np.zeros((512,512,3),dtype=np.float64)
    datapad = imgp
    
  
    print('datapad.shape',datapad.shape)
    Ny,Nx,Nc = datapad.shape
    



    ###没有mask,这里没有使用师兄的make_dsr_rgb函数。

    ###修改aloha_patch_rgb函数，该成没有mask的自己的函数，使用不同的切块方式，用余的函数，
    ###本函数的作用是取出想要的块，送进网络
    rimg = aloha_patch_rgb(datapad)

    # recon = recon * ( 1. - mask)  + dimg    ## DC
    recon=rimg
    print(recon.shape,recon.max(),recon.min())
    recon = np.clip(recon,0,1)
    
    save_img(recon,os.path.join('./results/Rec/', aaa))
    io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})
    print('图片数量：',aaa)
