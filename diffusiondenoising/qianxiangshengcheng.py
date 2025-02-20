#####qianxiangshengcheng前向生成

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
import scipy.io


img_path=r'/home/liuqg/wgj/diffu2/lzdata/wang0220/xueguantest60.png'
data = cv2.imread(img_path,0)
data = data / 255.

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




#scipy.io.savemat('qiangxiangy0.mat', mdict={'y': y0, })
scipy.io.savemat('512xueguansignal.mat',{'y':y0})
data=scipy.io.loadmat('512xueguansignal.mat')
yy=data['y']


yy=yy.tolist()
yy=matlab.double(yy)





recon=engine.backward2(yy)
recon=np.array(recon)
print(recon.shape)
print(recon)
recon=(recon-recon.min())/(recon.max()-recon.min())
#cv2.imshow('image', recon) 
cv2.imwrite('./wwresult/512xueguangsignal.png',255.*recon)

