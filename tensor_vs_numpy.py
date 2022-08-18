import numpy as np
import torch



# 默认类型
print("##############对于整形##############")
a=torch.tensor([4,6])
b= np.array([1,2,3])
print(a.dtype) # torch.int64
print(b.dtype) # int32

# 从 numpy 转 tensor时，numpy默认为int32，
# 而两种方法均保留的原类型，变成torch.int32，
# 而非tensor的默认类型torch.float32
c = torch.from_numpy(b) 
print(c.dtype) # torch.int64
d = torch.tensor(b)
print(d.dtype)

print("##############对于浮点型##############")

aa = torch.tensor([4.0,6.0])
bb = np.array([1.0,2.0,3.0])
print(aa.dtype) # torch.float32
print(bb.dtype) # float64


# 从 numpy 转 tensor时，numpy默认为float64，
# 而两种方法均保留的原类型，变成torch.float64，
# 而非tensor的默认类型torch.float32
cc = torch.from_numpy(bb) 
print(cc.dtype) # torch.float64
dd = torch.tensor(bb).float()
print(dd.dtype)





# # tensor 转 numpy
# dd = torch.from_numpy(a)
# c.dtype

print('end')


# c = torch.from_numpy(b)
# print(c.dtype)
# 可以看到，torch默认int型是64位的，numpy默认int型是32位的