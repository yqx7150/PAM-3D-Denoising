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
  ckpt_filename = "exp(old)/checkpoints/checkpoint_7.pth"
  
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
                                                        
#batch = next(eval_iter)#.float().cuda() # [1, 322, 192, 192]

#cv2.imwrite('ori.png',np.clip(batch[0][0] *255., 0, 255 ).numpy().astype('uint8'))



#data = batch[0].float().cuda()  # 1, 256, 256, 3
#hankle_data = batch[1].float().cuda() # 1, 322, 192, 192


#img = batch['image']._numpy()
#img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device)
#show_samples(img)

#mask = torch.ones_like(data)  # # 1, 256, 256, 3
#mask[:, :, :, 16:] = 0.
#print((batch * mask).max(),(batch * mask).min(),type((batch * mask)))
#assert 0

os.makedirs('./results/inpainter/', exist_ok=True)
os.makedirs('./results/Rec/', exist_ok=True)
os.makedirs('./results/hankle/', exist_ok=True)


def compute_mask(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array  
    
    
def make_dsr_rgb(dimg,Nimg,Nfir,data):

    Nshrink=Nimg-Nfir+1;  # 25
    if Nimg%2==0:
        hNimg=int(Nimg/2)
    else:
        hNimg=int((Nimg-1)/2)  # 18

    
    
        
    Ny,Nx,Nc=dimg.shape
    
    
    arr0 = np.pad(dimg[:,:,0],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr1 = np.pad(dimg[:,:,1],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr2 = np.pad(dimg[:,:,2],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    
    dimgp = np.concatenate((arr0[:,:,np.newaxis],arr1[:,:,np.newaxis],arr2[:,:,np.newaxis]),2)
    
    arr3 = np.pad(data[:,:,0],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr4 = np.pad(data[:,:,1],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr5 = np.pad(data[:,:,2],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    
    datap = np.concatenate((arr3[:,:,np.newaxis],arr4[:,:,np.newaxis],arr5[:,:,np.newaxis]),2)
    
    
    
    
    # plt.imshow(dimgp)
    # plt.show()
    
    vmask_arr = np.ones((256,256))
    vmask_arr1 = np.pad(vmask_arr[:,:],((hNimg,hNimg),(hNimg,hNimg)),'constant')    
    vmask = np.concatenate((vmask_arr1[:,:,np.newaxis],vmask_arr1[:,:,np.newaxis],vmask_arr1[:,:,np.newaxis]),2)  
    
    
    M1 = vmask
    M1 = np.reshape(M1.T,[M1.shape[0]*M1.shape[1]*M1.shape[2],1])
    vid = np.where(M1 == 1 )[0]
    
    mmask=np.zeros((dimg.shape[0],dimg.shape[1]))
    
    eNshrink=math.floor(Nshrink/2)
    hNshrink=math.floor(Nshrink/1)
    
    
    # 这里注意NX NY用的一样的
    if int(Nshrink/2)==0:
        # mmask(np.arange(eNshrink,Ny-eNshrink,hNshrink),np.arange(eNshrink,Ny-eNshrink,hNshrink))=1;
        # mmask(np.arange(eNshrink,Ny-eNshrink,hNshrink),Nx-eNshrink)=1;
        # mmask(Ny-eNshrink,np.arange(eNshrink,Ny-eNshrink,hNshrink))=1;
        assert 0
        mmask[Ny-eNshrink,Nx-eNshrink]=1;
    else:
        for iii in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
            
            for jjj in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
                mmask[iii,jjj]=1
        # 
        for aaa in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
            mmask[aaa,Nx-eNshrink-1]=1
        for bbb in np.arange(eNshrink,Ny-eNshrink-1,hNshrink):
            mmask[Nx-eNshrink-1,bbb]=1

        mmask[Ny-eNshrink-1,Nx-eNshrink-1]=1;
    
    # print(np.argwhere(mmask==1),np.argwhere(mmask==1).shape)#,eNshrink.dtype)
    
    mmask = np.pad(mmask,((hNimg,hNimg),(hNimg,hNimg)),'constant') 
    
    M2 = mmask
    M2 = np.reshape(M2.T,[M2.shape[0]*M2.shape[1],1])
    mid = np.where(M2 == 1 )[0] 
    

    # print(mid,mid)
    # assert 0
    
    return dimgp,vid,mid,datap


def aloha_patch_rgb(dimg,mask,mid,Nimg,Nfir,mu,muiter,datap):
    if Nimg%2==0:
        hNimg=int(Nimg/2)
    else:
        hNimg=int((Nimg-1)/2)  #18
    
    hNfir=round((Nfir-1)/2)
    Ny=dimg.shape[0]
    rimg=np.zeros_like(dimg)
    map_count=np.zeros_like(dimg)
    N=len(mid)
    
    
    
    arr0 = np.pad(mask[:,:,0],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr1 = np.pad(mask[:,:,1],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    arr2 = np.pad(mask[:,:,2],((hNimg,hNimg),(hNimg,hNimg)),'constant')
    
    maskp = np.concatenate((arr0[:,:,np.newaxis],arr1[:,:,np.newaxis],arr2[:,:,np.newaxis]),2)
    
    
    # print(np.argwhere(maskp[:,:,0]==1))
    
    
    Nc=dimg.shape[2]
    
    opts={'maxit':10000,'Zfull':1,'DoQR':1,'print':0,'est_rank':2}  
    
    for iter_num in range(N):
        ucur=mid[iter_num]#-1
        uy=ucur%Ny+1  # 31 
        ux=math.floor(ucur/Ny)+1  # 31 
       # print(mid)
       # print(ucur,ucur%Ny,uy,math.floor(ucur/Ny),ux)
       
        if Nimg%2==0:
            roiy=np.arange(uy-hNimg-1,uy+hNimg-1)   
            roix=np.arange(ux-hNimg-1,ux+hNimg-1)
            roiys=[]
            roixs=[]
            for cccc in np.arange(round(hNfir),round(roiy.shape[0]-hNfir)+1):
                # print(cccc)
                # print(roiy[int(cccc)])
                # assert 0
                roiys.append(roiy[int(cccc)])
            for dddd in np.arange(round(hNfir),round(roix.shape[0]-hNfir)+1):    
                roixs.append(roix[int(dddd)])

        else:
            roiy=np.arange(uy-hNimg-1,uy+hNimg)   
            roix=np.arange(ux-hNimg-1,ux+hNimg)
            roiys=[]
            roixs=[]
            for cccc in np.arange(hNfir,roiy.shape[0]-hNfir):
                # print(cccc)
                # print(roiy[int(cccc)])
                # assert 0
                roiys.append(roiy[int(cccc)])
            for dddd in np.arange(hNfir,roix.shape[0]-hNfir):    
                roixs.append(roix[int(dddd)])
                
            # roixs=roix(hNfir+1:end-hNfir)
        # rmask = np.zeros((roiy.shape[0],roix.shape[0]))   # (37, 37) float64  
        # print(rmask.shape,rmask.dtype)
        # assert 0
        
        size_data = [Nimg,Nimg,3]
        ksize= [Nfir,Nfir]
        
        rmask = maskp[roiy[0]:roiy[-1]+1,roix[0]:roix[-1]+1,:]

        
        
      
        k_w  = dimg[roiy[0]:roiy[-1]+1,roix[0]:roix[-1]+1,:]
        
        ori_data = datap[roiy[0]:roiy[-1]+1,roix[0]:roix[-1]+1,:]
        
        save_img(rmask,'rmask.png')
        save_img(k_w,'inpain.png')
        save_img(ori_data,'ori_data.png')
        
        psnr_zero = compare_psnr(255. * k_w, 255. * ori_data, data_range=255)
        ssim_zero = compare_ssim(k_w, ori_data, data_range=1,multichannel=True)
        
        
        rmask = torch.from_numpy(rmask).cuda()
        
        k_w = torch.from_numpy(k_w).cuda()
        
        ori_data = torch.from_numpy(ori_data).cuda()
        
        
        
    

        # write_Data("./results/result_zero.txt", str(iter_num)+'-'+aaa, {"psnr_zero":psnr_zero, "ssim_zero":ssim_zero})
        
        
        
        
        X = pc_inpainter(score_model, aaa, ckpt_filename, rmask, k_w, size_data, ksize,ori_data, iter_num,hNfir)
        
       
       
        
       
    
    
          
  
            

        
     
    

        

       
    
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
        
        rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] = rimg[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:]+ \
                                                            X[hNfir:int(X.shape[0]-hNfir)+1, hNfir:int(X.shape[1]-hNfir)+1, :]
        
        map_count[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] = map_count[roiys[0]:roiys[-1]+1,roixs[0]:roixs[-1]+1,:] + 1
    

        
        #assert 0
    
    
    
    # print(maskp,maskp.shape,maskp.dtype)
    id = np.argwhere(map_count==0)
    for num_id in range(len(id)):
        map_count[id[num_id][0], id[num_id][1], id[num_id][2]]=1


    rimg_n=np.divide(rimg, map_count)

    #save_img(rimg,'./rimg.png')
    #io.savemat('./rimg.mat',{'data':rimg})
    return rimg_n

def save_img(img, img_path):

    img = np.clip(img*255,0,255)

    cv2.imwrite(img_path, img)
    


def write_Data(filedir, model_num,dic1):
    #filedir="result.txt"
    with open(os.path.join(filedir),"a+") as f:#a+
        #f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.writelines(str(model_num)+' '+ str(dic1))
        f.write('\n')


psnr_all = []
ssim_all = []
mse_all=[]
nmse_all=[]

path='./lzdata/test_bedroom_256'

Nimg=64 #37;
Nfir=8  #13; 
mu = 1e1
muiter = 50

for aaa in sorted(os.listdir(path)):

    file_path = os.path.join(path, aaa)
       
    
    siat_input=cv2.imread(file_path)
    data = siat_input / 255.
    
    arr = np.ones((256,256))
    # mask = compute_mask(arr,rate=0.70)  # 欠采率
    
    # mask = io.loadmat('./mask/mask256_x5.mat')['mask']

    
    mask = io.loadmat('./mask/mask30.mat')['mask']

    

    mask = mask[:,:,np.newaxis]
    mask = np.concatenate((mask,mask,mask),2)
    
    # mask = np.ones_like(data)  #  256, 256, 3
    # mask[85:170, 85:170, :] = 0.

    
    
    save_img(mask, './results/mask.png')

    print('=====ratio: ', np.sum(mask)/(256*256*3))

    
    dimg = np.zeros(data.shape).astype(np.float64)
      
    dimg = data*mask  #np.random.uniform(0,1,(256,256,3))*mask # #
    
    # print(data,data.shape,data.dtype, data.max(),data.min())
    # print(dimg,dimg.shape,dimg.dtype, dimg.max(),dimg.min())
    # assert 0
    
    save_img(dimg, './results/inpainter/' + aaa)
    
    psnr_zero = compare_psnr(255. * dimg, 255. * data, data_range=255)
    ssim_zero = compare_ssim(dimg, data, data_range=1,multichannel=True)
    
    print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
    write_Data("./results/result_zero.txt", 'ratio_'+str(np.sum(mask)/(256*256*3)) + '_' + aaa, {"psnr_zero":psnr_zero, "ssim_zero":ssim_zero})
    
    
    Ny,Nx,Nc = dimg.shape
    dimgp,vid,mid,datap = make_dsr_rgb(dimg,Nimg,Nfir,data)
    

    
    
    rimg = aloha_patch_rgb(dimgp,mask,mid,Nimg,Nfir,mu,muiter,datap)
    
    M3 = np.reshape(rimg.T,[rimg.shape[0]*rimg.shape[1]*rimg.shape[2], 1]) 
    recon   = np.reshape(M3[vid],(Ny,Nx,Nc) ,order = 'F')
    recon = recon * ( 1. - mask)  + dimg
    
    print(recon.shape,recon.max(),recon.min())
    recon = np.clip(recon,0,1)
    
    save_img(recon,os.path.join('./results/Rec/', aaa))
    io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})
    
    psnr = compare_psnr(255.* recon, 255. * data, data_range=255)
    ssim = compare_ssim(recon, data, data_range=1,multichannel=True)
    print(aaa, ' PSNR:', psnr,' SSIM:', ssim)

    mse = compare_mse(data,recon)

    nmse =  np.sum((recon - data) ** 2.) / np.sum(data**2)
    
    write_Data("./results/Rec/result_last.txt", 'ratio_'+str(np.sum(mask)/(256*256*3))+ckpt_filename + '_' + aaa, {"psnr":psnr, "ssim":ssim, "mse":mse, "nmse":nmse})   
    

    psnr_all.append(psnr)
    ssim_all.append(ssim)
    mse_all.append(mse)

    nmse_all.append(nmse)

ave_psnr = sum(psnr_all) / len(psnr_all)
PSNR_std = np.std(psnr_all)       

ave_ssim = sum(ssim_all) / len(ssim_all)
SSIM_std = np.std(ssim_all)

ave_mse = sum(mse_all) / len(mse_all)
MSE_std = np.std(mse_all)

ave_nmse = sum(nmse_all) / len(nmse_all)
NMSE_std = np.std(nmse_all) 

write_Data("./results/Rec/result_mean.txt", 'ratio_'+str(np.sum(mask)/(256*256*3))  ,{"ave_psnr":ave_psnr, "PSNR_std":PSNR_std, "ave_ssim":ave_ssim, "SSIM_std":SSIM_std,"ave_mse":ave_mse,"MSE_std":MSE_std,"ave_nmse":ave_nmse,"NMSE_std":NMSE_std})

    
      
