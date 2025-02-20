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
  ckpt_filename = "/home/liuqg/wgj/diffusiondenoising/exp/checkpoints/checkpoint_17.pth"  ###现在用新训练的模型
  
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
corrector = LangevinCorrector #@param ["LangevinCorrector", "Anneb  ledLangevinDynamics", "None"] {"type": "raw"}
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

def full_pc(dimg,y0):

    k_w=dimg
    X,data=pc_inpainter(score_model,aaa,ckpt_filename,k_w,y0)
    return X,data


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

path='/home/liuqg/wgj/diffusiondenoising/testdata1'
file_path='/home/liuqg/wgj/diffusiondenoising/testdata1/test1/x_10_2 (2).png'
#根据当时的路径来

#path='./lzdata/wangguijun1'

for aaa in sorted(os.listdir(path)):

    file_path = os.path.join(path, aaa)
    good_input=cv2.imread(file_path,0)
    data = good_input
    #data = good_input ./ 255.
    #print(data.shape)
    ####修改代码666
    #datapad = np.zeros((512,512),dtype=np.float64)
    datapad = np.zeros((256,256),dtype=np.float64)
    datapad = data  ####这里就是测试的数据，展成了256
    dimg = sde.prior_sampling((1,256,256))  #注意这里。尺寸
    dimg=dimg.squeeze(0)   #原来有.permute(1,2,0)
    dimg=dimg.numpy()            #处理一下
    print(dimg.shape)
    #save_img(dimg, './results/inpainter/'+aaa)
    psnr_zero = compare_psnr(255. * dimg, 255. * datapad, data_range=255)    
    ssim_zero = compare_ssim(dimg, datapad, data_range=1,multichannel=True)  
    print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)


    ####先进行前向过程，得到y0，可以进行一个反变换
    
    #####matlab调用
    #######首个前向过程，得到y0###
    import matlab
    import matlab.engine               # import matlab引擎
    import matlab
    import matlab.engine
    import cv2
    import numpy as np
    import os.path
    import copy
    #666
    #data=mma*data
    img=data
    print(img.dtype)
    img= img.astype(np.float32)
    # cv2.imshow('image', img) 
    print(img.shape)
    # cv2.waitKey(0)
    print(img.dtype)
    print(type(img))
    img=img.tolist()
    img=matlab.double(img)
    engine = matlab.engine.start_matlab()  # 启动matlab engine
    sensor_data111=engine.forward2(img)
    sensor_data111=np.array(sensor_data111)
    #print(type(sensor_data111))
    #print(sensor_data111.max(),sensor_data111.min())
    #y0=(sensor_data111-sensor_data111.min())/(sensor_data111.max()-sensor_data111.min())
    y0=sensor_data111
    print(y0.shape)  
    y0=y0.tolist()
    y0=matlab.double(y0) 


    recon=engine.backward2(y0)

    recon=np.array(recon)

    print(recon.shape)
    print(recon)
    recon=(recon-recon.min())/(recon.max()-recon.min())
    #cv2.imshow('image', recon) 
    cv2.imwrite('./oneback/y01.png',255.*recon)



    recon,data=full_pc(dimg,y0)  #预测器校正器操作
    print(recon.shape,recon.max(),recon.min())
    recon=recon.cpu()
    recon=recon.numpy()
    recon=(recon-recon.min())/(recon.max()-recon.min())
    # recon = np.clip(recon,0,1)
    recon1=255.*recon
    print('#######397')
    print(recon1.shape)
    print(data.shape)
    recon = recon.squeeze(0)  #看看网络出来的是什么
    # recon = recon.transpose(2,1,0)
    #####顺序该一下。
    recon = recon.transpose(1,2,0)

    save_img(recon,'./recon1.png')
    save_img(data,'./data1.png')
    save_img(recon,os.path.join('./results/Rec/', aaa))
    io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})
    #666
    #good_img = data[128:384,128:384]
    #bad_img = recon[128:384,128:384]
    good_img = data
    bad_img = recon    
    cv2.imwrite('./wwresult/good.png',255.*good_img)
    cv2.imwrite('./wwresult/bad.png',255.*bad_img)
    recon=bad_img
    data=good_img
'''

    psnr = compare_psnr(255.* recon, 255. * data, data_range=255)
    ssim = compare_ssim(recon, data, data_range=1,multichannel=True)
    print(aaa, ' PSNR:', psnr,' SSIM:', ssim)
    mse = compare_mse(data,recon)
    nmse =  np.sum((recon - data) ** 2.) / np.sum(data**2)
    write_Data("./results/Rec/result_last.txt", 'model'+ckpt_filename + '_' + aaa, {"psnr":psnr, "ssim":ssim, "mse":mse, "nmse":nmse})
    
    psnr_all.append(psnr)
    ssim_all.append(ssim)
    mse_all.append(mse)
    nmse_all.append(nmse)
'''
ave_psnr = sum(psnr_all) / len(psnr_all)
PSNR_std = np.std(psnr_all)       

ave_ssim = sum(ssim_all) / len(ssim_all)
SSIM_std = np.std(ssim_all)

ave_mse = sum(mse_all) / len(mse_all)
MSE_std = np.std(mse_all)

ave_nmse = sum(nmse_all) / len(nmse_all)
NMSE_std = np.std(nmse_all) 

write_Data("./results/Rec/result_mean.txt", 'r'  ,{"ave_psnr":ave_psnr, "PSNR_std":PSNR_std, "ave_ssim":ave_ssim, "SSIM_std":SSIM_std,"ave_mse":ave_mse,"MSE_std":MSE_std,"ave_nmse":ave_nmse,"NMSE_std":NMSE_std})

    
      
