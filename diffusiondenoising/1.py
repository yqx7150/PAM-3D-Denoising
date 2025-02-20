# import matlab
# import matlab.engine               # import matlab引擎

# # 启动一个新的MATLAB进程，并返回Python的一个变量，它是一个MatlabEngine对象，用于与MATLAB过程进行通信。
# eng = matlab.engine.start_matlab() # 可以调用matlab的内置函数。                                  
# d = eng.multiplication_matlab(3,2) # 可以调用matlab写的脚本函数
# print('d', d, type(d))

#########@@@@@@@@@@@222222@#######
# import matlab
# import matlab.engine


# engine = matlab.engine.start_matlab()  # 启动matlab engine
# engine.hellomatlab(nargout = 0)



##################33333#############
import matlab
import matlab.engine               # import matlab引擎
import matlab
import matlab.engine
import cv2
import numpy as np
import os.path
import copy

# # 启动一个新的MATLAB进程，并返回Python的一个变量，它是一个MatlabEngine对象，用于与MATLAB过程进行通信。
# eng = matlab.engine.start_matlab() # 可以调用matlab的内置函数。                                  
# d = eng.UBP(nargout = 0) # 可以调用matlab写的脚本函数
# # print('d', d, type(d))
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

bb=engine.backward1(sensor_data111)
bb=np.array(bb)
print(bb.shape)   
bb=(bb-bb.min())/(bb.max()-bb.min())
cv2.imshow('image1', bb)
cv2.waitKey(0)
print('*******')
  
