import os
from scipy.io import loadmat,savemat
import numpy as np
import matplotlib.pyplot as plt
import cv2


def compute_mask(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array  

# arr = np.ones((256,256))
# mask = compute_mask(arr,rate=0.70)  # 欠采率




#mask = loadmat('wordmask_128.mat')['wordmask']
#ndarray=np.pad(mask,((64,64), (64,64) ),'constant', constant_values=(1, 1))  # (256, 256) 1 0

#mask = loadmat('linear_512.mat')['mask']
#print(mask.keys())

#ndarray=mask[128:384,128:384 ]


#mask = loadmat('curve_mask512.mat')['mask']
#print(mask.keys())
#ndarray=mask[180:180+256,100:100+256 ]


#mask = loadmat('block_mask512_0.5.mat')['app_mask']
#print(mask.keys())
#ndarray=mask[128:128+256,128:128+256 ]

mask = loadmat('NCU-NCU.mat')['mask']
print(mask.shape,mask.max(),mask.min())


cv2.imwrite('NCU-NCU.png', mask*255)
ndarray=mask

for i in range(ndarray.shape[0]):
    for j in range(ndarray.shape[1]):
        if ndarray[i][j] == 0 or ndarray[i][j] == 1:
            print()
        else:    
            print('!!!! error:', i, j,ndarray[i][j])
            assert 0
print('11111111111111', np.sum(mask)/(256*256), mask.shape,mask.dtype,mask.max(),mask.min())


# savemat('{}'.format('mask30'),{"mask":ndarray})            
            
#plt.imshow(ndarray, cmap='gray')
#plt.show()

#print(ndarray, ndarray.shape, ndarray.max(), ndarray.min())
#assert 0
