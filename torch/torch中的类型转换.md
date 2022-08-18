### 【Torch 日积月累】之tensor类数据的类型和设备转换

#### 1 参数形式：

`Tensor.to`(*args, **kwargs) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### 2 功能：依据参数，进行数据类型，所在设备的转换。

Performs Tensor dtype and/or device conversion. A [`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) and [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) are inferred from the arguments of `self.to(*args, **kwargs)`.

#### 使用说明与样例

##### 3.1使用方法一：仅改变数据类型

`to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)`

参数详解，下同：

- non_blocking=False，表示数据采用非阻塞方式，在训练时，这样可以和锁页内存pinned memory配合，加快训练速度。
- copy=False，这里的copy容易被混淆，它仅用于当dtype或device已经是目标类型时，是否需要采用深复制重新定义一个对象。而在更一般的情况，如果目标dtype和当前dtype不一致，则默认使用深复制，不提供修改接口（这样也比较安全，因为避免了高位转低位时造成的数据溢出丢失）

使用案例(验证参数copy的效果)：

```python
import numpy as np
import torch

print("############## 原始数据类型 ##############")
a = torch.randn(2, 2)  # 默认类型，Initially dtype=float32, device=cpu
print(a)
print(a.dtype)
print(hex(id(a)))
print(hex(id(a[0][0])))

print("############## 使用tensor.to 转换后 ##############")
b=a.to(torch.float64)
print(hex(id(b)))
print(hex(id(b[0][0])))
print(b)
print(b.dtype)

print("############## 使用tensor.to 转换，但保持原样数据类型，从而验证参数copy的作用 ##############")
c=a.to(torch.float32)
print(hex(id(a)))
print(hex(id(c)))
print(hex(id(a[0][0])))
print(hex(id(c[0][0])))
print(c)
print(a.dtype)
print(c.dtype)

print("############## 对照组，当 copy=True时的输出效果 ##############")
d=a.to(torch.float32, copy=True)
print(hex(id(a)))
print(hex(id(d)))
print(hex(id(a[0][0])))
print(hex(id(d[0][0])))
print(c)
print(a.dtype)
print(d.dtype)

print("############## 验证深复制 ##############")
b[0][0]=999.5
print(b)
print(b.dtype)
print(hex(id(b)))
print(a)
print(a.dtype)
print(hex(id(a)))
```

输出为

```python
############## 原始数据类型 ##############
tensor([[ 0.6689, -0.9043],
        [ 1.1141, -0.2554]])
torch.float32
0x24629323c70
0x24629323b80

############## 使用tensor.to 转换后 ##############
0x24629323b30
0x24629323ea0
tensor([[ 0.6689, -0.9043],
        [ 1.1141, -0.2554]], dtype=torch.float64)
torch.float64

############## 使用tensor.to 转换，但保持原样数据类型，从而验证参数copy的作用 ##############
0x24629323c70
0x24629323c70
0x24629323590
0x24629323590
tensor([[ 0.6689, -0.9043],
        [ 1.1141, -0.2554]])
torch.float32
torch.float32

############## 对照组，当 copy=True时的输出效果 ##############
0x24629323c70
0x2462932c590
0x2462932c450
0x2462932c220
tensor([[ 0.6689, -0.9043],
        [ 1.1141, -0.2554]])
torch.float32
torch.float32

############## 验证深复制 ##############
tensor([[ 9.9950e+02, -9.0432e-01],
        [ 1.1141e+00, -2.5539e-01]], dtype=torch.float64)
torch.float64
0x24629323b30
tensor([[ 0.6689, -0.9043],
        [ 1.1141, -0.2554]])
torch.float32
0x24629323c70
```



##### 3.2 使用方法二：改变数据类型和设备

`to(device, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) `

使用案例：

```python
cuda0 = torch.device('cuda:0')
a2 = torch.randn(2, 2)
print(a2)
print(a2.dtype)
print(hex(id(a2)))

b2 = a2.to(cuda0)
print(b2)
print(b2.dtype)
print(hex(id(b2)))

c2 = b2.to(cuda0, dtype=torch.float64)
print(c2)
print(c2.dtype)
print(hex(id(c2)))
```

输出为

```python
tensor([[-0.5810, -0.2508],
        [ 0.6758,  0.4693]])
torch.float32
0x246061fdc70
tensor([[-0.5810, -0.2508],
        [ 0.6758,  0.4693]], device='cuda:0')
torch.float32
0x2460621d400
tensor([[-0.5810, -0.2508],
        [ 0.6758,  0.4693]], device='cuda:0', dtype=torch.float64)
torch.float64
0x246065a70e0
```



##### 3.3使用方法三：将类型转换为与另一个other：tensor一致

`to(other, non_blocking=False, copy=False) `

other也是一个tensor，需要提前定义

使用案例：

```python
a3 = torch.randn(2,1)
other = torch.randn((3,4), dtype=torch.half, device='cuda:0')
print(a3)
print(a3.dtype)
print(hex(id(a3)))
print(other)
print(other.dtype)
print(hex(id(other)))

b3=a3.to(other)
print(b3)
print(b3.dtype)
print(hex(id(b3)))
```

输出为

```python
tensor([[-1.4872],
        [-1.9471]])
torch.float32
0x2465c7670e0
tensor([[ 0.6177,  1.3418, -0.5405, -1.5771],
        [ 0.6143, -1.0752,  0.1556,  0.7422],
        [-1.2363,  0.9189, -0.6006, -1.0938]], device='cuda:0',
       dtype=torch.float16)
torch.float16
0x2460621e450
tensor([[-1.4873],
        [-1.9473]], device='cuda:0', dtype=torch.float16)
torch.float16
0x24651d02180
```

##### 3.4 其他常用的别名

对于一个Tensor类的对象

`self.int()` is equivalent to `self.to(torch.int32)`

`self.float()` is equivalent to `self.to(torch.float32)`

#### Source code

coming soon

#### Reference

https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to

https://pytorch.org/docs/stable/generated/torch.Tensor.int.html

https://pytorch.org/docs/stable/tensors.html