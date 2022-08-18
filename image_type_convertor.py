import cv2
from PIL import Image
from matplotlib import image

import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from device_control import main_device

# 总结三种方法cv2, PIL, matplotlib读取图片，及其相互转换方法
# 同时总结了tensor类与ndarray，Image类的互转
DEVICE = main_device
cv2_img = cv2.imread('./test.png', flags=cv2.IMREAD_COLOR) # 输出为ndarray类
'''
第二个参数表示加载的图像是什么类型，支持常见的三个参数值。可以直接使用数字来代替（1，0，-1），默认值为IMREAD_UNCHANDED.

 IMREAD_UNCHANDED(<0) 表示加载原图，不做任何改变 , 包括alpha通道，可以直接写-1
 IMREAD_GRAYSCALE(0) 表示把原图作为灰度图像加载进来。
 IMREAD_COLOR(>0) 表示默认参数，读入一副彩色图片，忽略alpha通道
'''
pil_img = Image.open('./test.png')   # 输出为Image类
mp_img  = image.imread('./test.png') # 输出为ndarray类
print(pil_img.size) # 特殊：PIL中size格式为(w,h)，无c, 且未定义shape方法。


# print("############### Image --> ndarray ###################")
# print(pil_img.mode)
# pil_img1=pil_img.convert('RGB') #因为这里原始图片位深度是32，因此需要将RGBA转换为RGB模式 
# print(pil_img1.mode)
# pil_img2=np.array(pil_img1)
# print(pil_img2.shape)

# print("############### ndarray --> Image ###################")
# npimg=np.array([[[1, 140, 230,],
#                  [30, 150, 50],
#                  [160, 170, 80]],
#                 [[80, 81, 82],
#                  [96, 111, 33],
#                  [50, 195, 2]],
#                 [[77, 168, 196],
#                  [241, 156, 250],
#                  [39, 61, 253]]])

# print(npimg.shape)   # 这里对应着(h,w,c)的ndarray顺序              
# npimg1 = Image.fromarray(np.uint8(npimg)) # 这里务必要讲int32转换为uint8格式，防止出错
# print(npimg1.size) 

print("############### 当采用cv2读取图片时，如何改变通道顺序，以便后续处理 ###################")
print(cv2_img.shape) # (h,w,c)
print(cv2_img.dtype) # uint8
# resize到相同尺寸
# cv2_img = cv2.resize(cv2_img, (992, 992), interpolation=cv2.INTER_LINEAR)

# # 两种方法实现BGR to RGB
cv2_img_RGB = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
# cv2_img_RGB = cv2_img[:, :, ::-1].copy() # 这里需要采用深复制

# # 实现(h,w,c) to (c,h,w)
# cv2_img_RGB=cv2_img_RGB.transpose(2,0,1)


# # 两种方法实现mini-batch维度的扩充
cv2_img_RGB_with_batch = np.expand_dims(cv2_img_RGB, 0)
# cv2_img_RGB_with_batch = cv2_img_RGB[None,:]



# 两种方法实现ndarray to tensor，并转换成浮点类型（因为后续要计算偏微分）并将数据转移到指定设备上
# cv2_to_tensor = torch.from_numpy(cv2_img_RGB_with_batch).float() #推荐使用，采用浅复制，更加节约内存
cv2_to_tensor = torch.tensor(cv2_img_RGB_with_batch).float()

print(cv2_to_tensor.max())
print(cv2_to_tensor.min())
print(cv2_to_tensor.type())
print(cv2_to_tensor.dtype)
print(cv2_to_tensor.size())
print(cv2_to_tensor.shape)

# 附加，如果采用别的方法读取图片，例如PIL，imageio, skimage, matplotlib等等，
# 则需要依据图像数据类型，图像通道顺序，做出合理改变

print("############### 如何将模型输出的tensor形式数据恢复为 ndarray，显示并保存 ###################")







