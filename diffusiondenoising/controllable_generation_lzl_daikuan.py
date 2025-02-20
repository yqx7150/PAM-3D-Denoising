import os
from models import utils as mutils
import torch
import numpy as np
from sampling import (
    NoneCorrector,
    NonePredictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
import functools
import cv2
import math

# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io as io
#import matlab.engine

from lmafit_mc_adp_gpu import lmafit_mc_adp

# from lmafit_mc_adp_v2_numpy import lmafit_mc_adp


def write_Data(model_num, psnr, ssim):
    with open(os.path.join("./results/", "result.txt"), "a+") as f:
        f.writelines(
            str(model_num)
            + ","
            + "["
            + str(round(psnr, 2))
            + ","
            + str(round(ssim, 4))
            + "]"
        )
        f.write("\n")


def write_Data_pic(pictime, time, psnr, ssim):
    with open(os.path.join("./", "result" + str(pictime) + ".txt"), "a+") as f:
        f.writelines(str(time) + "," + str(round(psnr, 4)) + "," + str(round(ssim, 4)))
        f.write("\n")


def save_img(img, img_path):

    img = np.clip(img * 255, 0, 255)

    cv2.imwrite(img_path, img)


def write_zero_Data(model_num, psnr, ssim):
    filedir = "result_zero.txt"
    print(os.path.join("./results/"))
    with open(os.path.join("./results/", filedir), "a+") as f:  # a+
        f.writelines(
            str(model_num)
            + " "
            + "["
            + str(round(psnr, 2))
            + " "
            + str(round(ssim, 4))
            + "]"
        )
        f.write("\n")


def compute_mask(array, rate=0.2):
    """按照数组模板生成对应的 0-1 矩阵，默认rate=0.2"""
    zeros_num = int(array.size * rate)  # 根据0的比率来得到 0的个数
    new_array = np.ones(array.size)  # 生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0  # 将一部分换为0
    np.random.shuffle(new_array)  # 将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)  # 重新定义矩阵的维度，与模板相同
    return re_array


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def im2row(im, winSize):
    size = (im).shape
    out = torch.zeros(
        (
            (size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1),
            winSize[0] * winSize[1],
            size[2],
        ),
        dtype=torch.float64,
    ).to(device)
    count = -1
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            temp1 = im[
                x : (size[0] - winSize[0] + x + 1),
                y : (size[1] - winSize[1] + y + 1),
                :,
            ]
            #   temp3 = np.reshape(temp1.cpu(),[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')

            temp2 = reshape_fortran(
                temp1,
                [(size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), 1, size[2]],
            )

            #   print( '11111111111111' , (temp2.cpu() == temp3).all())
            #   assert 0

            out[:, count, :] = temp2.squeeze()  # MATLAB reshape

    return out


def row2im(mtx, size_data, winSize):
    size_mtx = mtx.shape  # (63001, 36, 8)
    sx = size_data[0]  # 256
    sy = size_data[1]  # 256
    sz = size_mtx[2]  # 8

    res = torch.zeros((sx, sy, sz), dtype=torch.float64).to(device)
    W = torch.zeros((sx, sy, sz), dtype=torch.float64).to(device)
    out = torch.zeros((sx, sy, sz), dtype=torch.float64).to(device)
    count = -1

    # aaaa = np.reshape(np.squeeze(mtx[:,count,:]).cpu(),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')
    # bbbb = reshape_fortran((mtx[:,count,:]).squeeze(),[sx-winSize[0]+1,sy-winSize[1]+1,sz])

    # print( '111111111',(aaaa == bbbb.cpu()).all())
    # assert 0

    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx - winSize[0] + x + 1, y : sy - winSize[1] + y + 1, :] = res[
                x : sx - winSize[0] + x + 1, y : sy - winSize[1] + y + 1, :
            ] + reshape_fortran(
                (mtx[:, count, :]).squeeze(),
                [sx - winSize[0] + 1, sy - winSize[1] + 1, sz],
            )
            W[x : sx - winSize[0] + x + 1, y : sy - winSize[1] + y + 1, :] = (
                W[x : sx - winSize[0] + x + 1, y : sy - winSize[1] + y + 1, :] + 1
            )

    # out = np.multiply(res,1./W)
    out = torch.mul(res, 1.0 / W)
    return out


def back_sh(Z, Known, data):

    lis2 = Z.flatten(order="F")
    for i in range(len(Known)):
        lis2[Known[i]] = data[i]
    lis3 = np.reshape(lis2, (Z.shape[0], Z.shape[1], Z.shape[2]), order="F")
    return lis3
#fuction_path="/home/lqg/LJB/林/光声/"
#eng=matlab.engine.start_matlab()
#eng.addpath(fuction_path)
def get_pc_inpainter(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
):
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
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_inpainter(
        model,
        k_w,
        data,
        good_input,
        detector,
        ):  # k_w纯噪图，data测试图片
        """Predictor-Corrector (PC) sampler for image inpainting.

        Args:
          model: A score model.
          data: A PyTorch tensor that represents a mini-batch of images to inpaint.
          mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
            and value `0` marks pixels that require inpainting.

        Returns:
          Inpainted (complete) images.
        """
        device = torch.device("cuda:0")
        with torch.no_grad():
            #print(data.shape, data.dtype, type(data))
            import numpy as np

            timesteps = torch.linspace(sde.T, eps, sde.N)

            x_input = k_w
            # x_input=x_input.transpose(2,0,1)####这里修改
            x_input = (
                torch.from_numpy(x_input)
                .to(torch.float32)
                .to(device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            data = (
                torch.from_numpy(data)
                .to(torch.float32)
                .to(device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            x_mean = x_input  # x_mean最开始为纯噪图
            z = data
            xx = data
            
 
            def toNumpy(tensor):
                return tensor.cpu().numpy()

            def toTensor(array):
                return torch.from_numpy(array).cuda()

            z = toNumpy(z).squeeze()
            xx = toNumpy(xx).squeeze()

            x1 = x_mean
            x2 = x_mean
            x3 = x_mean
            ssimmax = -np.inf
            bestitertion = None
            ##########加权最小二乘保真
            ssimbest = 0
            namereserved = None
            for i in range(2000,5000):
                print("============", i)
                t = timesteps[i].to(device)
                vec_t = torch.ones(x_input.shape[0], device=t.device) * t
                print('============',x_mean.shape, x_mean.dtype)
                x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
                x_mean = toNumpy(x_mean).squeeze()
                print('=====',x_mean.dtype)
                #p  rint(torch.max(x_mean), torch.min(x_mean))


                ####保真
                n = toNumpy(data).squeeze()
                print('===========',x_mean.shape,xx.shape,z.shape,n.shape)
                w = 1 / (1 * (np.exp(xx / 22000)))
                w = np.diag(w)
                hyper = 225
                sum_diff = xx - x_mean  #xx = data
                
                norm_diff = xx - n     #z=data
                x_new = z - (w *norm_diff  + 2 * hyper *sum_diff) / (2 * hyper + w)
                z = x_new + 0.1 * (x_new - xx)
                xx = x_new
                x_mean = xx
                x_mean = toTensor(x_mean).squeeze()

                x1, x2, x3, x_mean = corrector_update_fn(
                    x1, x2, x3, x_mean, vec_t, model=model
                )
                #x_mean = x_mean.to(torch.float32).to(device)
                x_mean = toNumpy(x_mean).squeeze()

                ####保真
                w = 1 / (1 * (np.exp(xx / 22000)))
                w = np.diag(w)
                hyper = 225
                sum_diff = xx - x_mean
        
                norm_diff = xx - n 
                x_new = z - (w*norm_diff+ 2 * hyper * sum_diff) / (2 * hyper + w)
                z = x_new + 0.1 * (x_new - xx)
                xx = x_new
                x_mean = xx
                x_mean = toTensor(x_mean)

                # 校正后的保真
                x_mean = x_mean.squeeze(0).squeeze(0)
                x_show = (x_mean - x_mean.min()) / (x_mean.max() - x_mean.min())
                x_mean = x_show

                x_show = x_show.to("cpu")
                x_show = np.array(x_show)
                x_show = x_show * 255.0
                cv2.imwrite("./output/" + str(i) + ".png", x_show)
                psnr = compare_psnr(x_show, good_input, data_range=255)
                ssim = compare_ssim(x_show, good_input, data_range=255)
                if ssim > ssimmax:
                    ssimmax = ssim
                    bestiteation = i
                print("psnr:", psnr, "ssim:", ssim,"ssimmax:",ssimmax,"bestiteration:",bestiteation)
                #write_Data_pic(pictime=savetimes, time=i, psnr=psnr, ssim=ssim)
                x_mean = x_mean.unsqueeze(0).unsqueeze(0)
                x_mean = x_mean.to(torch.float32)

            cv2.imwrite("./out_best30/" + str(bestiteation) + ".png", x_show)
            return x_mean
    return pc_inpainter


def get_pc_colorizer(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
):
    """Create a image colorization function based on Predictor-Corrector (PC) sampling.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
      corrector: A subclass of `sampb ling.Corrector` that represents a corrector algorithm.
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
    M = torch.tensor(
        [
            [5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
            [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
            [5.7735026e-01, 4.0824822e-01, -7.0710683e-01],
        ]
    )
    # `invM` is the inverse transformation of `M`
    invM = torch.inverse(M)

    # Decouple a gray-scale image with `M`
    def decouple(inputs):
        return torch.einsum("bihw,ij->bjhw", inputs, M.to(inputs.device))

    # The inverse function to `decouple`.
    def couple(inputs):
        return torch.einsum("bihw,ij->bjhw", inputs, invM.to(inputs.device))

    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def get_colorization_update_fn(update_fn):
        """Modify update functions of predictor & corrector to incorporate information of gray-scale images."""

        def colorization_update_fn(model, gray_scale_img, x, t):
            mask = get_mask(x)
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x, x_mean = update_fn(x, vec_t, model=model)
            masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
            masked_data = (
                masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
            )
            x = couple(decouple(x) * (1.0 - mask) + masked_data * mask)
            x_mean = couple(decouple(x) * (1.0 - mask) + masked_data_mean * mask)
            return x, x_mean

        return colorization_update_fn

    def get_mask(image):
        mask = torch.cat(
            [torch.ones_like(image[:, :1, ...]), torch.zeros_like(image[:, 1:, ...])],
            dim=1,
        )
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
            x = couple(
                decouple(gray_scale_img) * mask
                + decouple(
                    sde.prior_sampling(shape).to(gray_scale_img.device) * (1.0 - mask)
                )
            )
            timesteps = torch.linspace(sde.T, eps, 2000)
            for i in range(2000):
                t = timesteps[i]
                x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
                x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

            return inverse_scaler(x_mean if denoise else x)

    return pc_colorizer
