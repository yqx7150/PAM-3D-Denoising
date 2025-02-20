import os
from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
import cv2
import math
# from skimage.measure import compare_psnr,compare_ssim
import scipy.io as io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from lmafit_mc_adp_gpu import lmafit_mc_adp
#from lmafit_mc_adp_v2_numpy import lmafit_mc_adp

def write_Data(filedir, model_num,psnr,ssim):
    #filedir="result.txt"
    with open(os.path.join('./results',filedir),"a+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')

def save_img(img, img_path):

    img = np.clip(img*255,0,255)

    cv2.imwrite(img_path, img)
    
def write_zero_Data(model_num,psnr,ssim):
    filedir="result_zero.txt"
    with open(os.path.join('./results/',filedir),"a+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
    
def compute_mask(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array    


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def im2row(im,winSize):
  size = (im).shape
  out = torch.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=torch.float64).cuda()
  count = -1
  for y in range(winSize[1]):
    for x in range(winSize[0]):
      count = count + 1                 
      temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
    #   temp3 = np.reshape(temp1.cpu(),[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')

      temp2 = reshape_fortran(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]])

    #   print( '11111111111111' , (temp2.cpu() == temp3).all())
    #   assert 0
      
      out[:,count,:] = temp2.squeeze() # MATLAB reshape          
		
  return out
  
def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape #(63001, 36, 8)
    sx = size_data[0] # 256
    sy = size_data[1] # 256
    sz = size_mtx[2] # 8
    
    res = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()
    W = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()
    out = torch.zeros((sx,sy,sz),dtype=torch.float64).cuda()
    count = -1

    # aaaa = np.reshape(np.squeeze(mtx[:,count,:]).cpu(),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')
    # bbbb = reshape_fortran((mtx[:,count,:]).squeeze(),[sx-winSize[0]+1,sy-winSize[1]+1,sz])

    # print( '111111111',(aaaa == bbbb.cpu()).all())
    # assert 0
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + reshape_fortran((mtx[:,count,:]).squeeze(),[sx-winSize[0]+1,sy-winSize[1]+1,sz])  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    # out = np.multiply(res,1./W)
    out = torch.mul(res,1./W)
    return out
    

def back_sh(Z,Known,data):
    
   
    lis2 = Z.flatten(order='F')
    
    for i in range(len(Known)):
        lis2[Known[i]] = data[i]
    
    lis3 = np.reshape(lis2,(Z.shape[0],Z.shape[1],Z.shape[2]),order = 'F')    
    
    return lis3

    


    
    


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    An inpainting function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)



  def pc_inpainter(model, file_path, ckpt_filename, mask, k_w, size_data, ksize, data, iter_num,hNfir):
    """Predictor-Corrector (PC) sampler for image inpainting.

    Args:
      model: A score model.
      data: A PyTorch tensor that represents a mini-batch of images to inpaint.
      mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.

    Returns:
      Inpainted (complete) images.
    """
    with torch.no_grad():
      
      timesteps = torch.linspace(sde.T, eps, sde.N)
      
      
      file_name = str(iter_num)+'-'+file_path#.split('/')[-1]     
      
       

    
      #mask = np.ones_like(data)  #  256, 256, 3
      #mask[85:170, 85:170, :] = 0.
      
      
      
      
      
      '''
      arr = io.loadmat('./mask/curve_mask.mat')["mask"]
      mask = arr[:,:,np.newaxis]
      mask = np.concatenate((mask,mask,mask),2)
      '''
      
      
    
      
      #rmask_M1 = mask
      #rmask_M1 = np.reshape(rmask_M1.T,[rmask_M1.shape[0]*rmask_M1.shape[1]*rmask_M1.shape[2],1])
      #meas_id = np.where(rmask_M1!=0)[0]
      
      

      
      
      #size_data = [37,37,3]
      #ksize=[13,13]
      #wnthresh = 0.2  # 0.3best  0.6
      

      
      
      
      #cmtx_M1 = np.reshape(k_w.T,[k_w.shape[0]*k_w.shape[1]*k_w.shape[2],1])
      #meas = cmtx_M1[meas_id] 
      
      
      mask_cmtx   = im2row(mask,ksize)

      mask_cmtx = reshape_fortran(mask_cmtx,[mask_cmtx.shape[0],mask_cmtx.shape[1]*mask_cmtx.shape[2]])    # (62001, 192)
      
      
      hankel=im2row(k_w, ksize)
      
      size_temp = hankel.shape
      A = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])  # (3249, 192)  # 欠彩的hankel数据
      
      
      
      
      
      
      
      
      
      
      M1 = mask_cmtx
      M1 = torch.reshape(M1.T,[M1.shape[0]*M1.shape[1],1])
      Known = torch.where(M1 == 1 )[0]   # np.int64
      
      M2 = torch.reshape(A.T,[A.shape[0]*A.shape[1], 1])
        
      lma_data  = M2[Known]  
      
      
      
      
      #lma_data  = torch.from_numpy(lma_data).cuda()
      #Known  = torch.from_numpy(Known).cuda()
      
      
      
     
    
        
        
        #print('svd_input=====',svd_input)
        #print('1111Known',type(Known),Known.shape,Known,Known.dtype)
        #print('2222data',type(lma_data),lma_data.shape,lma_data,lma_data.dtype)
        #assert 0
       
        
      opts={'maxit':10000,'Zfull':1,'DoQR':1,'print':0,'est_rank':2}  
       
      ###奇异值分解
      #=============================================== SVD
      
      
      U,V ,_= lmafit_mc_adp(A.shape[0],A.shape[1],1,Known,lma_data,opts)
      
      
      #U=np.zeros((3249,67) ,dtype=np.float64)
      #V=np.zeros((67,192) ,dtype=np.float64)
      
        #U = U.cpu().numpy()
      #print(U,U.shape,U.dtype)  # float64
      #print(V,V.shape,V.dtype)  # float64
      #assert 0

        
      V = V.T       #.cpu().numpy()
      
      #U = U.cpu().numpy()  # (3249, 48) float64
      #V = V.cpu().numpy()
      
      L0=torch.zeros((U.shape[0],V.shape[0]) ,dtype=torch.float64).cuda() # (62001, 192)
      L = A - torch.mm(U,V.T) + L0  


      
      
      UVL_in = torch.mm(U,V.T)-L
      

      
      ans_1 = torch.zeros((16,192,192),dtype=torch.float64)  
      


      
      #ans_1 = np.array(A_temp,dtype=np.float64) 
      for i in range(16):#diu 49
        cut=UVL_in[192*i:192*(i+1)]
        ans_1[i,:,:]=cut  # (16, 192, 192)
        
      save_img(ans_1.cpu().numpy()[0,:,:], './results/hankle/'+file_name)
      
      
        
      x_input=ans_1.to(torch.float32).cuda().unsqueeze(0)
      

      x_mean=x_input  # [1, 16, 192, 192]
      

      
      x1 = x_mean
      x2 = x_mean
      x3 = x_mean 
      
      max_psnr = 0
      max_ssim = 0
      
      # Initial sample
      #x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask) #初始噪声
      mu = 10
      muiter = 50 #50
      
      r=U.shape[1]

      psnr_last=[1]

      for i in range(sde.N):
        print('============', i)
        t = timesteps[i].cuda()
        vec_t = torch.ones(x_input.shape[0], device=t.device) * t  #  ???
        
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
        

        
        #尺寸恢复
        #x_mean = x_mean.cpu().numpy() # (1, 16, 192, 192)           
        x_mean = x_mean.clone().detach().squeeze(0)
        #x_mean=x_mean.squeeze(0) # (16, 192, 192)   
        
        #save_img(x_mean[0,:,:], './results/Pred_322_192'+file_name)
        
               
        A_new=torch.zeros((16*192,192),dtype=torch.float64)
        for i in range(16): 
          A_new[192*i:192*(i+1),:]=x_mean[i,:,:]
        A_no=UVL_in[16*192:,:]
        P_output=torch.cat((A_new.cuda(),A_no),0) # (3249, 192)
        
        P_output = reshape_fortran(P_output,[P_output.shape[0],int(P_output.shape[1]/3),3])
        
        #P_output = np.reshape(P_output,[P_output.shape[0],int(P_output.shape[1]/3),3],order = 'F')  # (3249, 64, 3) float64
        
        
        
        
        kcomplex_h = row2im(P_output, size_data,ksize )  # (64, 64, 3) float64
        
        
        

        
        kcomplex_h = kcomplex_h * (1. - mask) + data * mask  # 保真 
        
        
        
        #####再乘H得hankel矩阵
    
        hankel=im2row(kcomplex_h, ksize)  # (3249, 64, 3) float64
        
        
      
        size_temp = hankel.shape
        Hx = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])  # (62001, 192)
        
        U1 = torch.mm(mu*(Hx+L), V )       
        U2 = torch.linalg.inv(torch.eye(r).to(torch.float64).cuda()+torch.mm(mu*V.T,V))
        U = torch.mm(U1,U2)  # (3249, 48) float64
        
        
        V1 = torch.mm(mu*((Hx+L).T), U )    
        V2 = torch.linalg.inv(torch.eye(r).to(torch.float64).cuda()+torch.mm(mu*U.T,U))
        V = torch.mm(V1,V2)  # (192, 48) float64
       
        L = Hx - torch.mm(U,V.T) + L
        
        
       

        UVL_in = torch.mm(U,V.T)-L   # (3249, 192) float64
        
        
      

      
        ans_1 = torch.zeros((16,192,192),dtype=torch.float64)  
  
        #ans_1 = np.array(A_temp,dtype=np.float64) 
        for i in range(16):#diu 49
          cut=UVL_in[192*i:192*(i+1)]
          ans_1[i,:,:]=cut  # (16, 192, 192) float64
        
        
         
        #save_img(ans_1[0,:,:], './results/before_correct_322_192_192'+file_name)

        x_mean=ans_1.to(torch.float32).cuda().unsqueeze(0)   

        
        ##======================================================= Corrector
        x1,x2,x3,x_mean = corrector_update_fn(x1,x2,x3,x_mean, vec_t, model=model)
        

        #尺寸恢复
        #x_mean = x_mean.cpu().numpy() # (1, 16, 192, 192)           
        x_mean = x_mean.clone().detach().squeeze(0)
        #x_mean=x_mean.squeeze(0) # (16, 192, 192)   
        
        #save_img(x_mean[0,:,:], './results/Pred_322_192'+file_name)
        
               
        A_new=torch.zeros((16*192,192),dtype=torch.float64)
        for i in range(16): 
          A_new[192*i:192*(i+1),:]=x_mean[i,:,:]
        A_no=UVL_in[16*192:,:]
        P_output=torch.cat((A_new.cuda(),A_no),0) #(3249, 192)
        
        
        
        P_output = reshape_fortran(P_output,[P_output.shape[0],int(P_output.shape[1]/3),3])  # float64
        

        
        
        kcomplex_h = row2im(P_output, size_data,ksize )  # float64
        
        
        

        
        kcomplex_h = kcomplex_h * (1. - mask) + data * mask  # 保真
        

        
        
        
        rec_Image = np.clip(kcomplex_h.cpu().numpy(), 0 ,1)
        
        
        
        
        #rec_Image = ( kcomplex_h - kcomplex_h.min() ) / ( kcomplex_h.max() - kcomplex_h.min() )
        #print(kcomplex_h.max(), kcomplex_h.min())#,kcomplex_h.dtype)
        #print(data.max(),data.min())#,data.dtype)
        #assert 0
        
        # PSNR
        
        rec_psnr_Image = rec_Image[hNfir:int(rec_Image.shape[0]-hNfir)+1, hNfir:int(rec_Image.shape[1]-hNfir)+1, :]
        data_psnr_Image = data.cpu().numpy()[hNfir:int(data.shape[0]-hNfir)+1, hNfir:int(data.shape[1]-hNfir)+1, :]
  
        psnr = compare_psnr(255.* rec_psnr_Image, 255. * data_psnr_Image, data_range=255)
        ssim = compare_ssim(rec_psnr_Image, data_psnr_Image, data_range=1,multichannel=True)
        #print(' PSNR:', psnr,' SSIM:', ssim)  
        write_Data("result_all.txt", file_name,psnr,ssim) 
        
        if max_ssim<=ssim:
          max_ssim = ssim
        if max_psnr<=psnr:
          max_psnr = psnr
          save_img(rec_psnr_Image,os.path.join('./results/'+ file_name))
          io.savemat(os.path.join('./results/'+ file_name +'.mat'),{'data':rec_psnr_Image})
          X_recon_patch = kcomplex_h.cpu().numpy()
          io.savemat(os.path.join('./results/'+ 'ori'+file_name +'.mat'),{'data':data_psnr_Image})
          #write_Data('checkpoint',max_psnr,ssim) 
          iii = 0
        
        if psnr > 100:
          break
        if (psnr - psnr_last[-1]) < 0:
          iii += 1
          if iii >= 100:
            break
        #else:
        #  iii = 0  

        
        #####再乘H得hankel矩阵
    
        hankel=im2row(kcomplex_h, ksize)
      
        size_temp = hankel.shape
        Hx = reshape_fortran(hankel,[size_temp[0],size_temp[1]*size_temp[2]])  # (62001, 192)
        
        U1 = torch.mm(mu*(Hx+L), V )       
        U2 = torch.linalg.inv(torch.eye(r).to(torch.float64).cuda()+torch.mm(mu*V.T,V))
        U = torch.mm(U1,U2)  # (3249, 48) float64
        
        
        V1 = torch.mm(mu*((Hx+L).T), U )    
        V2 = torch.linalg.inv(torch.eye(r).to(torch.float64).cuda()+torch.mm(mu*U.T,U))
        V = torch.mm(V1,V2)  # (192, 48) float64
       
        L = Hx - torch.mm(U,V.T) + L
        
        
       

        UVL_in = torch.mm(U,V.T)-L
      

      
        ans_1 = torch.zeros((16,192,192),dtype=torch.float64)  
  
        #ans_1 = np.array(A_temp,dtype=np.float64) 
        for i in range(16):#diu 49
          cut=UVL_in[192*i:192*(i+1)]
          ans_1[i,:,:]=cut  # (16, 192, 192)
        
        
         
        #save_img(ans_1[0,:,:], './results/before_correct_322_192_192'+file_name)

        x_mean=ans_1.to(torch.float32).cuda().unsqueeze(0)  
        psnr_last.append(psnr) 
        
        
        
      write_Data("result_best.txt", ckpt_filename + '_' + file_name,max_psnr,max_ssim)   

      return X_recon_patch #x_mean, max_psnr,max_ssim  # inverse_scaler(x_mean if denoise else x)

  return pc_inpainter


def get_pc_colorizer(sde, predictor, corrector, inverse_scaler,
                     snr, n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create a image colorization function based on Predictor-Corrector (PC) sampling.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates that the score-based model was trained with continuous time steps.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will start from `eps` to avoid numerical stabilities.

  Returns: A colorization function.
  """

  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                   [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                   [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  # `invM` is the inverse transformation of `M`
  invM = torch.inverse(M)

  # Decouple a gray-scale image with `M`
  def decouple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

  # The inverse function to `decouple`.
  def couple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_colorization_update_fn(update_fn):
    """Modify update functions of predictor & corrector to incorporate information of gray-scale images."""

    def colorization_update_fn(model, gray_scale_img, x, t):
      mask = get_mask(x)
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      x, x_mean = update_fn(x, vec_t, model=model)
      masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
      masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      x = couple(decouple(x) * (1. - mask) + masked_data * mask)
      x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
      return x, x_mean

    return colorization_update_fn

  def get_mask(image):
    mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                      torch.zeros_like(image[:, 1:, ...])], dim=1)
    return mask

  predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
  corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

  def pc_colorizer(model, gray_scale_img):
    """Colorize gray-scale images using Predictor-Corrector (PC) sampler.

    Args:
      model: A score model.
      gray_scale_img: A minibatch of gray-scale images. Their R,G,B channels have same values.

    Returns:
      Colorized images.
    """
    with torch.no_grad():
      shape = gray_scale_img.shape
      mask = get_mask(gray_scale_img)
      # Initial sample
      x = couple(decouple(gray_scale_img) * mask + \
                 decouple(sde.prior_sampling(shape).to(gray_scale_img.device)
                          * (1. - mask)))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
        x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_colorizer
