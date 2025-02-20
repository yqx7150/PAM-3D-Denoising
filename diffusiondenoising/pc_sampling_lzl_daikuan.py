from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np

# import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import controllable_generation_lzl_daikuan as controllable_generation
from utils import restore_checkpoint
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
from sampling import (
    ReverseDiffusionPredictor,
    LangevinCorrector,
    EulerMaruyamaPredictor,
    AncestralSamplingPredictor,
    NoneCorrector,
    NonePredictor,
    AnnealedLangevinDynamics,
)
import datasets
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import math
import scipy.io as io
sns.set(font_scale=2)
sns.set(style="whitegrid")


# @title Load the score-based model
sde = "VESDE"  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == "vesde":
    from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
    ckpt_filename = "exp/checkpoints/checkpoint_50.pth"  # 2000 迭代的最好模型
    # ckpt_filename ="/home/lqg/LZL/diffu1/exp/checkpoints/checkpoint_11.pth"  # 1000 迭代的最好模型
    print(ckpt_filename)
    if not os.path.exists(ckpt_filename):
        print("!!!!!!!!!!!!!!" + ckpt_filename + " not exists")
        assert False
    config = configs.get_config()
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )
    sampling_eps = 1e-5
    batch_size = 1  # @param {"type":"integer"}
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    random_seed = 0  # @param {"type": "integer"}
    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)
    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    # @title PC inpainting

    predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.21  # 0.07#0.075 #0.16 #@param {"type": "number"}
    n_steps = 1  # @param {"type": "integer"}
    probability_flow = False  # @param {"type": "boolean"}

    pc_inpainter = controllable_generation.get_pc_inpainter(
        sde,
        predictor,
        corrector,
        inverse_scaler,
        snr=snr,
        n_steps=n_steps,
        probability_flow=probability_flow,
        continuous=config.training.continuous,
        denoise=True,
    )

    os.makedirs("./results/inpainter/", exist_ok=True)  ####原来没有1
    os.makedirs("./results/Recnew/", exist_ok=True)
    os.makedirs("./results/hankle/", exist_ok=True)

    def full_pc(dimg, data,  good_input, detector):
        X = pc_inpainter(
            model=score_model,
            k_w=dimg,
            data=data,
            good_input=good_input,
            detector=detector,
        )
        return X



    bad_input = "/home/liuqg/wgj/diffu5/血管/snr20grey/032.jpg"                  #"具体图片路径"
    good_path = "/home/liuqg/wgj/diffu5/血管/snrnonisegrey/032.jpg"                  #"具体图片路径"
    save_all_path = "./save_all"              #"图片存放文件夹"
    detector=512
    print(bad_input,'\n',good_path)
    bad_input = cv2.imread(bad_input, 0)
    data = bad_input / 255.0
    good_input = cv2.imread(good_path, 0)

    dimg = sde.prior_sampling((1, 256, 256))  # 注意这里。尺寸
    dimg = dimg.squeeze(0)  # 原来有.permute(1,2,0)
    dimg = dimg.numpy()  # 处理一下
    recon = full_pc(
        dimg=dimg,data= data,good_input=good_input,detector=detector
    )  # 预测器校正器操作

    recon = recon.cpu().numpy()
    recon = (recon - recon.min()) / (recon.max() - recon.min())
    recon = recon.squeeze(0).transpose(1, 2, 0).squeeze(2)
    psnr = compare_psnr(255.0 * recon, 255.0 * good_input, data_range=255)
    ssim = compare_ssim(recon, good_input, data_range=1, multichannel=True)
    cv2.imwrite(save_all_path,recon)
    write_Data_for(psnr, ssim)
    print(aaa, " PSNR:", psnr, " SSIM:", ssim)
    mse = compare_mse(good_input, recon)
    nmse = np.sum((recon - good_input) ** 2.0) / np.sum(good_input**2)

