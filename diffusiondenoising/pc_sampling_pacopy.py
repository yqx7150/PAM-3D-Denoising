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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# from skimage.measure import compare_psnr,compare_ssim


import math
import scipy.io as io



# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  ckpt_filename = "exp/checkpoints/checkpoint_10.pth"  ###现在用新训练的模型
  
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
                                                        


os.makedirs('./results/inpainter/', exist_ok=True)  ####原来没有1
os.makedirs('./results/Rec/', exist_ok=True)
os.makedirs('./results/hankle/', exist_ok=True)

def full_pc(dimg,datap):

    k_w=dimg
    ori_data=datap
    X=pc_inpainter(score_model,aaa,ckpt_filename,k_w,ori_data)
    return X


def save_img(img, img_path):

    img = np.clip(img*255,0,255)    ##最小值0, 最大值255

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

#path='./lzdata/wgj_testnew8'

path='./lzdata/wangguijun1'
for aaa in sorted(os.listdir(path)):

    file_path = os.path.join(path, aaa)
    good_input=cv2.imread(file_path,0)
    
    data = good_input / 255.
    print(data.shape)
    
    datapad = np.zeros((512,512),dtype=np.float64)
    datapad = data  ####这里就是测试的数据，展成了512
    dimg = sde.prior_sampling((1,512,512))  #注意这里。他的尺寸不太对
    dimg=dimg.squeeze(0)   #原来有.permute(1,2,0)
    dimg=dimg.numpy()            #不知道这个要不要处理一下？
    print(dimg.shape)
    save_img(dimg, './results/inpainter/'+aaa)
    psnr_zero = compare_psnr(255. * dimg, 255. * datapad, data_range=255)    
    ssim_zero = compare_ssim(dimg, datapad, data_range=1,multichannel=True)  
    print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)

    recon=full_pc(dimg,datapad)  ###经hankel后pc操作
    print(recon.shape,recon.max(),recon.min())
    recon=recon.cpu()
    recon=recon.numpy()
    recon=(recon-recon.min())/(recon.max()-recon.min())
    # recon = np.clip(recon,0,1)
    recon1=255.*recon
    print('#######397')
    print(recon1.shape)
    print(recon1)
    recon = recon.squeeze(0)  #这里还不知道要不要压缩，看看网络出来的是什么
    # recon = recon.transpose(2,1,0)
    #####顺序该一下。
    recon = recon.transpose(1,2,0)

    save_img(recon,'./154.png')
    save_img(recon,os.path.join('./results/Rec/', aaa))
    io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})
    
    psnr = compare_psnr(255.* recon, 255. * datapad, data_range=255)
    ssim = compare_ssim(recon, datapad, data_range=1,multichannel=True)
    print(aaa, ' PSNR:', psnr,' SSIM:', ssim)
    mse = compare_mse(datapad,recon)
    nmse =  np.sum((recon - datapad) ** 2.) / np.sum(datapad**2)
    # write_Data("./results/Rec/result_last.txt", 'ratio_'+str(np.sum(mask)/(256*256*3))+ckpt_filename + '_' + aaa, {"psnr":psnr, "ssim":ssim, "mse":mse, "nmse":nmse})
    
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

# write_Data("./results/Rec/result_mean.txt", 'ratio_'+str(np.sum(mask)/(512*512*3))  ,{"ave_psnr":ave_psnr, "PSNR_std":PSNR_std, "ave_ssim":ave_ssim, "SSIM_std":SSIM_std,"ave_mse":ave_mse,"MSE_std":MSE_std,"ave_nmse":ave_nmse,"NMSE_std":NMSE_std})

    
      
