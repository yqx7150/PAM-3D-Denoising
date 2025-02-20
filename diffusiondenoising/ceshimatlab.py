####计算psnr
'''


import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

mma=np.ones((256,256))
mma=np.pad(mma,((128,128),(128,128)),'constant')

'''
'''
img_path=r'/home/liuqg/wgj/diffu1/02.png'
img = cv2.imread(img_path,0)
print(img.dtype)
img= img.astype(np.float32)
img1=mma*img
cv2.imwrite('./wwresult/03.png',img1)
'''
'''


img_path1=r'/home/liuqg/桌面/pretest/wwresult/03.png'

img_path2=r'/home/liuqg/桌面/pretest/wwresult/128_14_0.6.png'

good_img = cv2.imread(img_path1,0)
good_img = good_img / 255.

good_img = good_img[128:384,128:384]
print(good_img.shape)

bad_img = cv2.imread(img_path2,0)
bad_img = bad_img / 255.
bad_img = bad_img[128:384,128:384]
cv2.imwrite('./wwresult/good.png',255.*good_img)
cv2.imwrite('./wwresult/bad.png',255.*bad_img)



psnr = compare_psnr(255.* bad_img, 255. * good_img, data_range=255)
#ssim = compare_ssim(bad_img, good_img, data_range=1,multichannel=True)
ssim = compare_ssim(bad_img, good_img, multichannel=True)
print(' PSNR:', psnr,' SSIM:', ssim)



'''


####调用测试
#######首个前向过程，得到y0###
import matlab
import matlab.engine               # import matlab引擎
import matlab
import matlab.engine
import cv2
import numpy as np
import os.path
import copy


img_path=r'/home/liuqg/wgj/diffu2/lzdata/wang0220/xueguantest60.png'
data = cv2.imread(img_path,0)

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
cv2.imwrite('./wwresult/512xueguangsignalto64_3.png',255.*recon)

