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


#####matalab测试###
import matlab
import matlab.engine               # import matlab引擎
import matlab
import matlab.engine
import cv2
import numpy as np
import os.path
import copy


'''
img_path=r'/home/lqg/wgj/pa/k-Wave/02.png'
img = cv2.imread(img_path,0)
print(img.dtype)
img= img.astype(np.float32)
# cv2.imshow('image', img) 
print(img.shape)
# cv2.waitKey(0)
print(img.dtype)
img=img.tolist()
img=matlab.double(img)

engine = matlab.engine.start_matlab()  # 启动matlab engine

sensor_data111=engine.forward1(img)
sensor_data111=np.array(sensor_data111)
print(type(sensor_data111))
print(sensor_data111.max(),sensor_data111.min())
sensor_data111=(sensor_data111-sensor_data111.min())/(sensor_data111.max()-sensor_data111.min())
print(sensor_data111.shape)   
# cv2.imshow('image', sensor_data111)


	sensor_data111=sensor_data111.tolist()
sensor_data111=matlab.double(sensor_data111)
 
img_path=r'/home/lqg/wgj/pa/k-Wave/xueguang1.png'
img1 = cv2.imread(img_path,0)
img1=img1.tolist()
img1=matlab.double(img1)
net=engine.gradupcompute(img1,sensor_data111)
bb=net
bb=np.array(bb)
print(bb.shape)   
bb=(bb-bb.min())/(bb.max()-bb.min())
cv2.imshow('image1', bb)
cv2.waitKey(0)
print('*******')

#########
'''
# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  ckpt_filename = "exp1/checkpoints/checkpoint_3.pth"  ###现在用新训练的模型
  
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


def compute_mask(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array  
    
    

###分块
###分块###
def get_cut_data(data,flag):
    i=0
    j=0
    if flag<=7:
        i=0
        j=flag
    elif (flag<=15)and(flag>=7):
        i=1
        j=flag-8
    elif (flag<=23)and(flag>=15):
        i=2
        j=flag-16
    elif (flag<=31)and(flag>=23):
        i=3
        j=flag-24
    elif (flag<=39)and(flag>=31):
        i=4
        j=flag-32
    elif (flag<=47)and(flag>=39):
        i=5
        j=flag-40
    elif (flag<=55)and(flag>=47):
        i=6
        j=flag-48
    else:
        i=7
        j=flag-56

    data_s=np.zeros([64,64,3])
    data_s = data[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :]




    return data_s


###块拼接###

def get_recover_data(data,flag,sample_data):
    i = 0
    j = 0
    if flag <= 7:
        i = 0
        j = flag
    elif (flag <= 15) and (flag >= 7):
        i = 1
        j = flag - 8
    elif (flag <= 23) and (flag >= 15):
        i = 2
        j = flag - 16
    elif (flag <= 31) and (flag >= 23):
        i = 3
        j = flag - 24
    elif (flag <= 39) and (flag >= 31):
        i = 4
        j = flag - 32
    elif (flag <= 47) and (flag >= 39):
        i = 5
        j = flag - 40
    elif (flag <= 55) and (flag >= 47):
        i = 6
        j = flag - 48
    else:
        i = 7
        j = flag - 56

    #不知道这里是不是改对的，原设置了dtype
    # rec_data=np.zeros([3, 512, 512])
    rec_data = np.zeros([512, 512,3])

    rec_data=sample_data


    # rec_data[:,i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]=data
    rec_data[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = data
    print(data.shape)
    print(i*64,(i+1)*64,j*64,(j+1)*64)
    print(i,j)

    return rec_data


###重写一个
def full_pc(dimg,datap):

    k_w=dimg
    ori_data=datap
    X=pc_inpainter(score_model,aaa,ckpt_filename,k_w,ori_data)


    return X



  
    # for flag in range(64):  ###ceshi yuan 64
    #     k_w=get_cut_data(dimg,flag)
    #     ori_data=get_cut_data(datap,flag)
    #     save_img(k_w,'inpain.png')
    #     save_img(ori_data,'ori_data.png')
    #     psnr_zero = compare_psnr(255. * k_w, 255. * ori_data, data_range=255)
    #     ssim_zero = compare_ssim(k_w, ori_data, data_range=1, multichannel=True)
    #     k_w = torch.from_numpy(k_w).cuda()
    #     ori_data = torch.from_numpy(ori_data).cuda()
    #     X = pc_inpainter(score_model, aaa, ckpt_filename, k_w, size_data, ksize,ori_data,flag)
    #     print('########  466')
    #     print(X.shape)
    #     print('#####468  ceshi  3,64,64')
    #     plt.imshow(X)
    #     plt.show()
    #     # # assert 0
    #     print('####478')
    #     print(flag)


    #     rec_data=get_recover_data(X,flag,rec_data)


    # return rec_data





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

path='./lzdata/wangguijun1'
#path='./lzdata/wgj_testnew8'

'''
## 图像填充，填充成512*512*3
def pad_image(dimg, dup, ddown, dleft, dright):
    arr0 = np.pad(dimg[:, :, 0], ((dup, ddown), (dleft, dright)), 'constant')
    arr1 = np.pad(dimg[:, :, 1], ((dup, ddown), (dleft, dright)), 'constant')
    arr2 = np.pad(dimg[:, :, 2], ((dup, ddown), (dleft, dright)), 'constant')

    imgp = np.concatenate((arr0[:, :, np.newaxis], arr1[:, :, np.newaxis], arr2[:, :, np.newaxis]), 2)

    return imgp


def pad_image(dimg, dup, ddown, dleft, dright):
    arr0 = np.pad(dimg[:, :, 0], ((dup, ddown), (dleft, dright)), 'constant')
    arr1 = np.pad(dimg[:, :, 1], ((dup, ddown), (dleft, dright)), 'constant')
    arr2 = np.pad(dimg[:, :, 2], ((dup, ddown), (dleft, dright)), 'constant')

    imgp = np.concatenate((arr0[:, :, np.newaxis], arr1[:, :, np.newaxis], arr2[:, :, np.newaxis]), 2)

    return imgp

'''

## 图像填充，填充成512*512*3
def pad_image(dimg, dup, ddown, dleft, dright):
    arr0 = np.pad(dimg[:, :], ((dup, ddown), (dleft, dright)), 'constant')
    imgp=arr0   

    return imgp



for aaa in sorted(os.listdir(path)):

    file_path = os.path.join(path, aaa)
       
    
    good_input=cv2.imread(file_path,0)

    '''
    #############
    img=good_input
    print(img.dtype)
    img= img.astype(np.float32)
    # cv2.imshow('image', img) 
    print(img.shape)
    # cv2.waitKey(0)
    print(img.dtype)
    img=img.tolist()
    img=matlab.double(img)

    engine = matlab.engine.start_matlab()  # 启动matlab engine

    sensor_data111=engine.forward1(img)
    sensor_data111=np.array(sensor_data111)
    print(type(sensor_data111))
    print(sensor_data111.max(),sensor_data111.min())
    sensor_data111=(sensor_data111-sensor_data111.min())/(sensor_data111.max()-sensor_data111.min())
    print(sensor_data111.shape)   
    # cv2.imshow('image', sensor_data111)


    sensor_data111=sensor_data111.tolist()
    sensor_data111=matlab.double(sensor_data111)
 
    #img_path=r'/home/lqg/wgj/pa/k-Wave/xueguang1.png'
    #img1 = cv2.imread(img_path,0)
    #img1=img1.tolist()
    #img1=matlab.double(img1)
    net=engine.backward1(sensor_data111)
    bb=net
    bb=np.array(bb)
    print(bb.shape)   
    bb=(bb-bb.min())/(bb.max()-bb.min())
    cv2.imshow('image1', bb)
    cv2.waitKey(0)
    print('*******')
    assert 0

    #########
    '''

    #good_input=cv2.imread('./lzdata/bad8/1.jpg')




    data = good_input / 255.
    # data = siat_input
    print("#################################  yu  ##############")
    print('data.shape',data.shape)
    
    nx,ny = data.shape    ## nx=375, ny=500, nz=3
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


    imgp = pad_image(data, pup, pdown, pleft, pright)    ## 512*512*3
    
    
    datapad = np.zeros((512,512),dtype=np.float64)
    datapad = imgp   ####这里就是测试的数据，展成了512
	

    print("#################################  yu  ##############")
    print('datapad.shape',datapad.shape)

    ###########************* 在这里修改，添加 pad，depad
    
    arr = np.ones((512,512))


    dimg = sde.prior_sampling((1,3,512,512))
    dimg=dimg.squeeze(0).permute(1,2,0)
    dimg=dimg.numpy()

    # print("#################################  yu  ##############")
    # print('dimg.shape',dimg.shape)
    

    # dimg= np.ascontiguousarray(dimg)
#########z362####
   
    dimg=dimg / 255.
    #########
#
#     dimg = sde.prior_sampling((1,3,512,512))
#     dimg=dimg.squeeze(0).permute(1,2,0)
#     dimg=dimg.numpy()
    print(dimg.shape)
    save_img(dimg, './results/inpainter/'+aaa)
    
    #psnr_zero = compare_psnr(255. * dimg, 255. * data, data_range=255)    ##**##
    #psnr_zero = compare_psnr(255. * dimg, 255. * datapad, data_range=255)    
    
    #ssim_zero = compare_ssim(dimg, data, data_range=1,multichannel=True)    ##**##
    #ssim_zero = compare_ssim(dimg, datapad, data_range=1,multichannel=True)  
    
    #print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
    # write_Data("./results/result_zero.txt", 'ratio_'+ '_' + aaa, {"psnr_zero":psnr_zero, "ssim_zero":ssim_zero})
    
    
    Ny,Nx,Nc = dimg.shape
    #dimgp,vid,mid,datap = make_dsr_rgb(dimg,Nimg,Nfir,data)    ##**##
    # dimgp,vid,mid,datap = make_dsr_rgb(dimg,Nimg,Nfir,datapad)#####自己分块 不需要标注
    

    
    
    # rimg = aloha_patch_rgb(dimgp,mask,mid,Nimg,Nfir,mu,muiter,datap)####重写一个
    rimg=full_pc(dimg,datapad)  ###经hankel后pc操作


    # recon=rimg+datapad #####这里原来有，但是是不因该的,没有什么意义
    recon=rimg

    print(recon.shape,recon.max(),recon.min())
    recon=recon.cpu()
    recon=recon.numpy()
    recon=(recon-recon.min())/(recon.max()-recon.min())
    # recon = np.clip(recon,0,1)
    recon1=255.*recon
    print('#######397')
    print(recon1.shape)
    print(recon1)
    recon = recon.squeeze(0)
    # recon = recon.transpose(2,1,0)
    #####顺序该一下。
    recon = recon.transpose(1,2,0)

    save_img(recon,'./154.png')
    save_img(recon,os.path.join('./results/Rec/', aaa))
    io.savemat(os.path.join('./results/Rec/',aaa +'.mat'),{'data':recon})
    
    #psnr = compare_psnr(255.* recon, 255. * data, data_range=255)    ##**##
    #ssim = compare_ssim(recon, data, data_range=1,multichannel=True)    ##**##
    
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

    
      
