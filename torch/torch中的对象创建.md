## 【torch日积月累】之tensor的常用函数使用要义

### 1 tensor的常见创建方法

#### 1.1 torch.tensor(data) 

当定义一个tensor对象时，我们一般会常用`torch.tensor(ndarray or list)`方法，但是①该方法仅能够对输入的参数进行深复制`copy`（会创建新的storage区，与原data不共享内存），且②并不会保留`autograd history`。

> 官方文档原话：torch.tensor constructs a tensor with no autograd history (also known as a “leaf tensor”)

众所周知，python中的list对象，因为采用了引用语义的原因。普遍存在着赋值和深浅拷贝的问题。而在torch框架中，虽然不存在引用语义的问题，但是却要频繁的进行高维张量的计算。为了节省内存的考虑，也衍生出类似于浅拷贝的方法（共享内存），避免深拷贝。同时为了方便计算偏微分，也需要考虑autograd history的问题。为此，以下的三个函数常常作为`torch.tensor()`的替代函数。

##### 1.1.1 torch.from_numpy()

功能相对简单，将输入的ndarray转换为tensor（并保留原始数据类型）

参数形式：

`torch.from_numpy`(*ndarray*) → Tensor

**仅支持**可读的*ndarray*作为输入。并且creates a tensor that shares storage(共享内存地址) 。这里的共享存储区的机制，也是一种避免深拷贝的方法，有利于节省内存空间。The returned tensor is not resizable.

##### 1.1.2 torch.as_tensor()

preserves autograd history and avoids copies where possible。这里的意思是，只有在已满足dtype和device（或者为None）才会避免copy，这里间接调用了`Tensor.to()`的method。相比上面的`from_numpy`，该方法更加灵活而普适，它强制保留autograd history且允许更改dtype和device（若更改，则必须copy）。

参数形式：

`torch.as_tensor`(*data*, *dtype=None*, *device=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

data支持ndarray，list，tuple，标量，tensor等等作为输入

**情况1**：指定了新的dtype和device时，则必然会copy。

**情况2**：如果数据是标量，list等普通的对象，因为输入数据的 dtype和device不存在，且未被指定（==None），该函数会将其转换为tensor的默认数据类型。这里存在当高位数据被转换为地位数据时，数据可能丢失的问题。例如python的64浮点数被转换为torch中默认的32位浮点数时：

```python
# 使用 torch.as_tensor()初始化, but for list
a = [9999999999999999999999999999999999999999999999.99999,6.5]
t = torch.as_tensor(a) # 发生copy
print(t,t.dtype)
t[0] = -1
print(a,type(a))
print(t,t.dtype)
```

输出为：

```python
tensor([   inf, 6.5000]) torch.float32
[1e+46, 6.5] <class 'list'>
tensor([-1.0000,  6.5000]) torch.float32
```

这里64位转32位时，超过了32位浮点数的最大范围，因此转化过后变为正无穷。由此也可以看出，在进行科研开发时，不太推荐使用list对象，而更加推荐使用ndarray，则可以很好地规避这样的问题（如需使用，则必须手动自动目标的dtype为64位，不然torch会默认转换成torch.float32）。

下面的转而用numpy的对照情况，可以发现这里并没有发生精度丢失问题。因为采用了

```python
# 使用 torch.as_tensor()初始化, but for ndarray
a = np.array([9999999999999999999999999999999999999999999999.99999,6.5])
t = torch.as_tensor(a) 
print(t,t.dtype)
print(a,a.dtype)

```

输出为：

```python
tensor([1.0000e+46, 6.5000e+00], dtype=torch.float64) torch.float64
[1.0e+46 6.5e+00] float64
```

**情况3**：If data is a NumPy array (an ndarray) with the same dtype and device then a tensor is constructed using [`torch.from_numpy()`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy).当输入data为ndarray，且不需要变换 dtype 和 device 时（指None或符合要求），则会隐式的调用[`torch.from_numpy()`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy)。只有这种情况，内存共享机制才会被触发。在实测中，发现了如果输入data为tensor时，与输入ndarray出现了类似的效果。

##### 1.1.3 torch.clone()

 参数说明：该方法既可以视为函数，又可以视为method。

 `torch.clone`(*input*, ***, *memory_format=torch.preserve_format*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

该函数的主要意义在于，copy了新的存储区（内存不共享），并且保持了与input之间的梯度传播关系。

> 特别注意：This function is differentiable, so gradients will flow back from the result of this operation to `input`. To create a tensor without an autograd relationship to `input` see [`detach()`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html#torch.Tensor.detach).

这段话的意思是，该函数是可微的，在`input.requires_grad=True`的情况下，对output的反向传播结果，也传播（叠加）到input的上。也即input和output处在同一个计算图当中。

##### 1.1.4 Tensor.detach()和Tensor.requires_grad_()

当我们在训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播。切断后的部分网络结构默认的`requires_grad=False`，如果这部分网络需要单独训练，则可以通过就地method `Tensor.requires_grad_()`进行更改，使其具有梯度。

**需要注意的是，detach()方法依然是共享内存的，并未创建新的存储区。这一点和clone()是相反的。clone()不共享内存但保留梯度流动；而detach()共享内存但切断了梯度流动。下面是一个示例**

```python
a = torch.tensor(1.0, requires_grad=True)
b = a.clone().detach() # 割裂a和b为两个网络
print(a.data_ptr(), b.data_ptr())
print(a.requires_grad, b.requires_grad)
# (True False)  # 两者的requires_grad都是True
b.requires_grad_()

c = a * 2
c.backward()
print(a.grad)# a的梯度值为tensor(2.)

d = b * 3
d.backward()
print(b.grad) # b的梯度值为tensor(3.)

print(a.grad) # a的梯度值为tensor(2.)
```

事实上，这个示例中同时使用了`clone()`和`detach()`，该操作和`torch.tensor()`是等价的。也即`b = a.clone().detach()`等价于`b=torch.tensor(a)`。此外，`b=torch.tensor(a,requires_grad=True)` 也等价于 `b=a.clone().detach().requires_grad_()`。



##### 1.1.5 torch.tensor()的参数形式和用法

```python
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

data: 可以是list、tuple、ndarray、标量、tensor（此时相当于`tensor.clone()`）等等。

- dtype: 返回张量所需的数据类型。默认:如果为None，则从data推断数据类型。
- device: 返回张量的所需设备CPU或GPU。默认:如果为None，则使用默认张量类型的当前设备
- requires_grad: 是否计算autograd，默认为False
- pin_memory: 如果设置了，返回的张量将在固定内存中分配。只对CPU张量有效。默认为False

```python
# 使用 torch.tensor()初始化
a = torch.tensor([[1, -1], [1, -1]])
b = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
c = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
d = torch.tensor([[1, -1], [1, -1]], dtype=torch.int32)
print(a) 
print('dtype of a is',a.dtype)
print('id of a is',a.data_ptr(),end='\n\n')

print(b)
print('dtype of b is',b.dtype)
print('id of b is',b.data_ptr(),end='\n\n')

print(c)
print('dtype of c is',c.dtype)
print('id of c is',c.data_ptr(),end='\n\n')

print(d)
print('dtype of d is',d.dtype)
print('id of d is',d.data_ptr())

# 创建一个空的tensor
print(torch.tensor([]))
# Output:
# tensor([])

```

输出为：

```python
tensor([[ 1, -1],
        [ 1, -1]])
dtype of a is torch.int64
id of a is 3232800068608

tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
dtype of b is torch.int32
id of b is 3232800062208

tensor([[1, 2, 3],
        [4, 5, 6]])
dtype of c is torch.int64
id of c is 3232796620032

tensor([[ 1, -1],
        [ 1, -1]], dtype=torch.int32)
dtype of d is torch.int32
id of d is 3232495737088

tensor([])
```

需要辨析的是，torch.tensor()是一个函数，与torch.Tensor不同，torch.Tensor是类的实例化。

- torch.Tensor(data)是将输入的data转换成torch.FloatTensor类型（默认float32类型）的对象, 而torch.tensor(data)是依据于data的类型或者dtype
- torch.Tensor()可以创建一个空的FloatTensor，torch.tensor()创建会出错，最少也要给个空的列表 []

#### 2.2 torch.sparse_coo_tensor()

参数形式：

```python
torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, requires_grad=False)
```

创建一个稀疏tensor,格式为coo类型（非零元素的坐标形式）。

稀疏tensor是tensor中数值为0的元素数目远远多于非0元素的数目，并且非0元素分布无规律

反之，非0元素占大多数，称为稠密tensor

indices：非零元素的坐标；values：非零元素的值

```python
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype = torch.float32)
torch.sparse_coo_tensor(i, v, [2, 4])

tensor(indices=tensor([[0, 1, 1],
                       [2, 0, 2]]),
       values=tensor([3., 4., 5.]),
       size=(2, 4), nnz=3, layout=torch.sparse_coo)


torch.sparse_coo_tensor(i, v, dtype = torch.float64, device = torch.device('cuda:0'))


tensor(indices=tensor([[0, 1, 1],
                   [2, 0, 2]]),
   values=tensor([3., 4., 5.]),
   device='cuda:0', size=(2, 3), nnz=3, dtype=torch.float64,
   layout=torch.sparse_coo)

```



#### 2.4 torch.as_strided()

```python
torch.as_strided(input, size, stride, storage_offset=0)
1
```

根据现有的tensor及步长来创建一个可视的tensor视图

- input: 输入的tensor
- size: 指定大小，需要指定行和列
- stride: 指定步长，也是需要指定行和列
- storage_offset: 输出张量的偏移量

```python
x = torch.randn(3, 3)
print(x)
12
tensor([[ 1.5352, -1.5436,  0.3728],
        [-0.9212, -0.2614, -1.0020],
        [-0.9214,  0.7106, -0.0114]])
123
t = torch.as_strided(x, (2, 2), (1, 2))
print(t)
12
tensor([[ 1.5352,  0.3728],
        [-1.5436, -0.9212]])
12
```

这里的参数(2, 2)很好理解，就是输出2行2列；而(1, 2)是指定步长的行和列，t的每一行从x中取值，因为行步长为1，取了1.5352，-1.5436放在t中每一行的开头；t的每一列也从给x中取值，第一列的值已经确定，第二列的值从x中按列步长2来取值，1.5352后两步是0.3728，-1.5436后两步是-0.9212

```python
# 再试一下输出3*3的，步长为（1，3），结果看着是转置的效果-_-
t = torch.as_strided(x, (3, 3), (1, 3))
print(t)
123
tensor([[ 1.5352, -0.9212, -0.9214],
        [-1.5436, -0.2614,  0.7106],
        [ 0.3728, -1.0020, -0.0114]])
123
# 还有个storage_offset参数，控制行步长是否偏移，默认为0
#
t = torch.as_strided(x, (2, 2), (1, 2), 1)
print(t)
1234
tensor([[-1.5436, -0.9212],
        [ 0.3728 -0.2614]])
12
```

#### 2.6 torch.zeros()

```python
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

返回一个张量，其标量值为0，形状由变量size定义

```python
torch.zeros(2, 3)
tensor([[0., 0., 0.],
        [0., 0., 0.]])

torch.zeros(5)
tensor([0., 0., 0., 0., 0.])

```

#### 2.7 torch.zeros_like()

```python
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
1
```

返回一个张量，其标量值为0，与输入的大小相同。torch.zeros_like(input)等价于torch.zeros(input.size()， dtype=input.dtype,layout=input.dtype,device= input.device).

```python
input = torch.empty(2, 3)
torch.zeros_like(input)
12
tensor([[0., 0., 0.],
        [0., 0., 0.]])
12
```

#### 2.8 torch.ones()

```python
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

返回一个张量，该张量填充标量值1，形状由变量size定义。

```python
torch.ones(2, 3)
1
tensor([[1., 1., 1.],
        [1., 1., 1.]])
12
torch.ones(5)
1
tensor([1., 1., 1., 1., 1.])
1
```

#### 2.9 torch.ones_like()

```python
torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
1
```

返回一个张量，其标量值为1，与输入的大小相同。torch.ones_like(input)等价于torch.ones(input.size()， dtype=input.dtype,layout=input.dtype,device= input.device).

```python
input = torch.empty(2, 3)
torch.ones_like(input)
12
tensor([[1., 1., 1.],
        [1., 1., 1.]])
12
```

#### 2.10 torch.arrnge()

```python
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

创建一个通过步长step间隔，大小为(end-start)/step的一维张量

start默认值为0

step默认值为1

end需要指定

```python
#只指定end
a = torch.arange(5)
a
123
tensor([0, 1, 2, 3, 4])
1
# 指定start,end
b = torch.arange(1, 4)
b
123
tensor([1, 2, 3])
1
# 指定start,end, step，但输出的张量不包含end
torch.arange(1, 3, 0.5)
12
tensor([1.0000, 1.5000, 2.0000, 2.5000])
1
# 不能整除时，输出的张量也不包含end
torch.arange(1, 3, 0.3)
12
tensor([1.0000, 1.3000, 1.6000, 1.9000, 2.2000, 2.5000, 2.8000])
1
```

#### 2.11 torch.range()

```python
torch.range(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

此函数已弃用，并将在未来版本中删除，建议用上个torch.arange()

#### 2.12 torch.linspace()

```python
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

创建一个均匀间隔steps，大小为(end-start)/steps的一维张量

```python
torch.linspace(3, 10, steps = 5)

tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])

```

#### 2.13 torch.logspace()

```python
torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 
1
```

创建一个起始b a s e s t a r t base^{start}basestart均匀间隔steps，结束为b a s e e n d base^{end}baseend的一维张量

```python
torch.logspace(start = -10, end = 10, steps = 5)
1
tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
1
```

#### 2.14 torch.eye()

```python
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

返回一个二维张量，对角线上为1，其他地方为0。

n: 张量函数

m：默认等于n

```python
torch.eye(3)
1
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
123
torch.eye(3, 2)
1
tensor([[1., 0.],
        [0., 1.],
        [0., 0.]])
123
```

#### 2.15 torch.empty()

```python
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
1
```

返回一个包含未初始化数据的张量。张量的形状是由变量size定义的

```python
torch.empty(2)
1
tensor([1.0653e-38, 1.0469e-38])
1
# size可以是元组
torch.empty(2, 3)
12
tensor([[0.0000e+00, 4.1292e-05, 1.0500e-08],
        [1.6894e-07, 4.2969e-05, 1.6853e+22]])
12
# size也可以是list
torch.empty([2, 3])
12
tensor([[8.4113e+20, 1.0569e+21, 1.0573e-05],
        [1.6521e-07, 1.3543e-05, 6.6523e+22]])
12
```

#### 2.16 torch.empty_like()

```python
torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
1
```

返回与输入大小相同的未初始化张量。torch.emptys_like(input)等价于torch.emptys(input.size()， dtype=input.dtype,layout=input.dtype,device= input.device).

输入的必须是Tensor

```python
i = torch.tensor([2., 3.])
t = torch.empty_like(i)
print(t)
t.dtype
1234
tensor([-5.7296e+35,  4.5916e-41])

torch.float32
123
```

#### 2.17 torch.empty_strided()

```python
torch.empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False)
1
```

返回一个包含未初始化数据的张量。张量的形状和步幅分别由变量size和stride定义。

torch.empty_strided(size, stride)等价于torch.empty(size).as_strided(size, stride).

```python
a = torch.empty_strided((2, 3), (1, 2))
a
12
tensor([[6.6460e+22, 4.1726e+21, 5.1205e-11],
        [4.1292e-05, 8.3112e+20, 7.5034e+28]])
12
a.stride()
1
(1, 2)
1
a.size()
1
torch.Size([2, 3])
1
```

#### 2.18 torch.full()

```python
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
1
```

创建一个用fill_value填充的size张量。张量的dtype是从fill_value推断出来的。

```python
torch.full((2, 3), 3.14)
1
tensor([[3.1400, 3.1400, 3.1400],
        [3.1400, 3.1400, 3.1400]])
12
```

#### 2.19 torch.full_like()

```python
torch.full_like(input, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
1
```

返回一个张量，其大小与填充了fill_value的输入相同。

torch.full_like(input, fill_value)等价于torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device).

#### 2.20 torch.quantize_per_tensor()

```python
torch.quantize_per_tensor(input, scale, zero_point, dtype)
1
```

将浮点张量转换为具有给定尺度和零点的量化张量。

```python
a = torch.quantize_per_tensor(torch.tensor([-1.0, 3.0, 1.0, 2.0]), 0.1, 10, torch.quint8)
a
12
tensor([-1.,  3.,  1.,  2.], size=(4,), dtype=torch.quint8,
       quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=10)
12
a.dtype
1
torch.quint8
1
```

量化是指用于执行计算并以低于浮点精度的位宽存储张量的技术，主要是一种加速推理的技术，并且量化算子仅支持前向传递。
*量化参考*：

- https://pytorch.apachecn.org/docs/1.4/88.html
- https://zhuanlan.zhihu.com/p/144025236

这个函数是[pytorch](https://so.csdn.net/so/search?q=pytorch&spm=1001.2101.3001.7020)量化中的静态量化，给定了scale和zero_point，将32位浮点数量化成了8位定点数

还有动态量化：torch.quantization.quantize_dynamic

```python
# 一般量化后的模型带入运算时，需要用dequantize()函数解除量化，重新变为float32
a.dequantize().dtype
12
torch.float32
1
```

#### 2.21 torch.quantize_per_channel()

```python
torch.quantize_per_channel(input, scales, zero_points, axis, dtype)
1
```

将浮点张量转换为具有给定尺度和零点的每个通道量化张量。

这个和上一个对比，上一个按Tensor量化，这个函数按channel量化

参数部分，除了需要指定scale，zero_point，dtype，还需指定per_channel的维度

```python
# 这里的scale和zero_point就是一维张量，而且要和input的size匹配
x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8)
123
tensor([[-1.,  0.],
        [ 1.,  2.]], size=(2, 2), dtype=torch.quint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.1000, 0.0100], dtype=torch.float64),
       zero_point=tensor([10,  0]), axis=0)
12345
```

#### 2.22 torch.dequantize(tensor)

这个函数在2.20以经用过了，就是反量化的，把量化的张量重新转换成float32

#### 2.23 torch.complex()

```python
torch.complex(real, imag, *, out=None)
```

构造一个复张量，其实部等于real，虚部等于imag。

real必须时float或double,imag和real保持一致

```python
# 如果real和imag都是float32，输出为complex64
real = torch.tensor([1, 2], dtype = torch.float32)
imag = torch.tensor([3, 4], dtype = torch.float32)
z = torch.complex(real, imag)
print(z)
z.dtype
123456
tensor([1.+3.j, 2.+4.j])


torch.complex64
1234
```

#### 2.24 torch.polar()

```python
torch.polar(abs, angle, *, out=None)
1
```

构造一个complex张量，其元素为笛卡尔坐标，对应于绝对值为abs、夹角为angle的极坐标

out=abs⋅cos(angle)+abs⋅sin(angle)⋅j

abs: 必须为float或double

angle: 和abs保持一致

```python
# 如果abs和angle都是float64，输出为complex128
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
z = torch.polar(abs, angle)
print(z)
z.dtype
123456
tensor([ 6.1232e-17+1.0000j, -1.4142e+00-1.4142j], dtype=torch.complex128)


torch.complex128
1234
```













#### 



### 2 常用的随机采样器

#### 2.1  torch.rand和torch.rand_like
参数形式：

`torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)` → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

功能: 依据参数size, 在[0, 1)范围内返回**均匀分布**的随机张量。

另外，`torch.rand_like(input)`用于返回与`input`尺寸一致的张量。其参数形式如下：

`torch.rand_like`(*input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

这里的`out`参数，指的是为输出赋值为一个别名（占用同一个地址），并未创造新的对象。

#### 2.2  torch.randn和torch.randn_like

`torch.randn`(*size, *, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

功能：按照**标准正态分布**$\text{out}_{i} \sim \mathcal{N}(0, 1)$返回特定尺寸的随机张量

`torch.randn_like`可类比，同上。

`torch.randn_like`(*input*, *, *dtype=None*, *layout=None*, *device=None*, *requires_grad=False*, *memory_format=torch.preserve_format*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### 2.3  torch.randint和torch.randint_like

`torch.randint`(*low=0*, *high*, *size*, *\**, *generator=None*, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

功能：在(low,high)区间内返回特定size的张量，默认dtype是torch.int64。

`torch.randint_like`(*input*, *low=0*, *high*, *\**, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*, *memory_format=torch.preserve_format*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

例子：

```python
torch.randint(3, 5, (3,))
torch.randint(10, (2, 2))
torch.randint(3, 10, (2, 2))
```

#### 2.4 torch.randperm

`torch.randperm`(*n*, *, *generator=None*, *out=None*, *dtype=torch.int64*, *layout=torch.strided*, *device=None*, *requires_grad=False*, *pin_memory=False*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

功能：返回从[0,n)范围内的扰动整数序列

Returns a random permutation of integers from `0` to `n - 1`.

例子：

```python
torch.randperm(4)
# output： tensor([2, 1, 0, 3])
```

#### 2.5 随机数管理器torch.Generator

PyTorch 通过 `torch.Generator` 类来管理随机数的生成。有两种方式可以实现Generator类的实例化。分别是：基于 [torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).xxx 的默认实例化，和 基于Generator 的自定义实例。这两种方法均可调用 manual_seed() 以及 initial_seed(), 前者相当于使用默认的 Generator 实例去调用相应方法；而后者则是采用自定义的实例，可以单独设定种子和随机数的生成。下面以默认实例化方法为例：

```python
import torch
# 设置随机数种子
torch.manual_seed(666)
print(torch.initial_seed()) # 查看随机数种子 666

g_2 = torch.manual_seed(0)
print(g_2.initial_seed()) # 0

# 获取默认的 Generator 实例
g_1 = torch.default_generator
print(g_1.initial_seed()) # 0

# 结果为 True
print(g_1 is g_2) # True
```

下面以自定义实例化方法为例：

```python
# 手动创建的随机数生成器
g3 = torch.Generator()
print(g3.initial_seed()) # 67280421310721
g3.manual_seed(1)
print(g3.initial_seed()) # 1
```

##### （1）种子设定管理

其中，对于种子的设定，官方文档建议采用尽可能大的种子数值，且种子数值的0 bit和1 bit的数量要尽可能的保持平衡。原文如下：

> It is recommended to set a large seed, i.e. a number that has a good balance of 0 and 1 bits. Avoid having many 0 bits in the seed.
>
> Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]. Otherwise, a RuntimeError is raised. Negative inputs are remapped to positive values with the formula 0xffff_ffff_ffff_ffff + seed.

此外，该类也提供一个自动设置种子的方法，该方法从`std::random_device`或当前时间获取不确定的随机数并将其用作生成器的种子

> Gets a non-deterministic random number from std::random_device or the current time and uses it to seed a Generator.

例子如下：

```python
# 手动创建随机数生成器
G = torch.Generator(device='cuda')
G.manual_seed(1)
print(G.initial_seed()) # 1

G.seed()
print(G.initial_seed()) # 3710802736670458
```

##### （2）设备位置管理

Generator 类包含一个关键字参数`device='cpu'`, 会区分 CPU 与 GPU 两种设备, 默认为 CPU 类型

```python
# 使用默认的随机数生成器
torch.manual_seed(1)

# 结果 tensor([0, 4, 2, 3, 1])
print(torch.randperm(5))
print(torch.default_generator.device) # cpu

# 手动创建随机数生成器
G = torch.Generator(device='cuda')
G.manual_seed(1)


# 结果 tensor([0, 4, 2, 3, 1])
print(torch.randperm(5, generator=G, device='cuda'))
print(G.device) # cuda
```

##### （3）实例的深拷贝

通过`get_state()`和`set_state(old_state)`联合使用，可以实现实例的深拷贝，例子如下：

```python
kk = torch.Generator()
kk.seed()
print(kk.initial_seed()) # 576724579900700

a=kk.get_state() # 查看该类的状态码
# print(G.get_state())
print('the id of kk is: ',id(kk)) # the id of kk is:  2128861923824

g_cpu_other = torch.Generator()
print(g_cpu_other.initial_seed()) # 67280421310721
g_cpu_other.set_state(kk.get_state())

print(g_cpu_other.initial_seed()) # 576724579900700

print('the id of g_cpu_other is: ',id(g_cpu_other)) # the id of g_cpu_other is:  2128861593168
```

可以发现，两个实例占用了不同的地址。另外，对于默认类型的Generator实例，则需要改用[`get_rng_state`](https://pytorch.org/docs/stable/generated/torch.get_rng_state.html#torch.get_rng_state)和[`set_rng_state`](https://pytorch.org/docs/stable/generated/torch.set_rng_state.html#torch.set_rng_state)。`get_state()`和`set_state(old_state)`对默认实例无效。

#### 2.6  torch.bernoulli

参数形式：

`torch.bernoulli`(input, *, *generator*=None, *out*=None) → Tensor

功能：基于输入的概率，做为伯努利分布的参数，决定每个位置的输出是0或1

> Draws binary random numbers (0 or 1) from a Bernoulli distribution.

用法举例：

```python
a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
print(a)
print(torch.bernoulli(a, generator=kk))
# kk需要自己定义，或者取默认None，可以参考2.5节
```

输出为：

```python
tensor([[0.1652, 0.2628, 0.6705],
        [0.5896, 0.2873, 0.3486],
        [0.9579, 0.4075, 0.7819]])
tensor([[1., 0., 0.],
        [1., 0., 0.],
        [1., 1., 1.]])
```

另，补充一个就地伯努利采样的方法，参数形式如下：

`Tensor.bernoulli_`(*p=0.5*, *, *generator=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

这里的p可以是一个概率的标量，也可以是一个包含与tensor实例相同size的多维张量。

```python
a1=torch.ones(3, 3)
b1=torch.empty(3, 3).uniform_(0, 1)
a1.bernoulli_(b1)
print(a1)

c1=torch.randint_like(a1,10,device='cuda')
c1.bernoulli_(0.99)
print(c1)
```

输出为：

```python
tensor([[1., 0., 1.],
        [0., 0., 0.],
        [0., 0., 1.]])
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
```

#### 2.7 torch.normal

参数形式：正态分布采样器

`torch.normal`(*mean=0.0*, *std*=1.0, size=None, *, *generator=None*, *out=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

示例代码：

```python
a2=torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
print(a2) # 一般情况

b2=torch.normal(mean=250, std=torch.arange(1., 6.,device='cuda'))
print(b2) # 简洁形式

c2=torch.normal(mean=torch.arange(1., 6.,device='cuda'), std=0.1)
print(c2)  # 这里和上一条，表明了输出的tensor的设备位置受到了参数的影响。

d2=torch.normal(mean=torch.arange(1., 13.).reshape(3,4), std=torch.arange(1., 13.).reshape(1,12))
print(d2) # 这里表明了mean和std允许有不同的shape，但element数量要保持一致。且输出的shape与mean一致

e2 = torch.normal(0, 88, size=(1, 4))
print(e2)
```

输出为：

```python
tensor([ 3.0170,  2.7382,  2.4615,  3.6176,  4.7924,  5.6880,  6.6683,  7.7693,
         9.1002, 10.0412])

tensor([251.3551, 251.3911, 252.0018, 245.8633, 248.5067], device='cuda:0')

tensor([0.8146, 2.0670, 2.9554, 3.9749, 5.0211], device='cuda:0')

tensor([[ 1.1248,  3.7747, -1.0588,  6.6303],
        [ 1.1274,  5.7302, 17.1167,  8.3153],
        [ 8.6236,  2.6557, -9.5730,  6.1679]])

tensor([[  21.9903,    8.2003,  -41.6389, -126.4552]])
```

#### 2.8 torch.poisson

参数形式：

`torch.poisson`(*input*, *generator=None*) → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

示例代码：

```python
rates = torch.rand(4, 4, device='cuda') * 5  # rate parameter between 0 and 5
torch.poisson(rates)
```

输出为：

```python
tensor([[0., 3., 1., 1.],
        [2., 2., 5., 0.],
        [0., 3., 5., 0.],
        [1., 1., 2., 5.]], device='cuda:0')
```

#### 2.9 torch.multinomial()

参数形式：

`torch.multinomial`(*input*,  *num_samples*, *replacement=False*, ***, *generator=None*, *out=None*) → LongTensor

功能：对每一行的索引值，按照其数值的权重大小确定概率，进行*num_samples*次不放回的随机取样。

> 引用一个来自知乎的解释：input可以看成一个权重张量，每一个元素的值代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取完之前，该元素不会被取到。

其中参数replacement为True，表示是有放回，False则无放回。此外，如input是多行张量。则有如下规则：

> If `input` is a vector, `out` is a vector of size `num_samples`.
>
> If `input` is a matrix with m rows, `out` is an matrix of shape ($ m \times \text{num_samples}$)($m×\text{num_samples}$).

示例代码：

```python
weights = torch.tensor([0, 5, 3, 0], dtype=torch.float) # create a tensor of weights
print(torch.multinomial(weights, 2))  
print(torch.multinomial(weights, 4))

print(torch.multinomial(weights, 4, replacement=True))
```

输出为：

```python
tensor([1, 2])
tensor([1, 2, 0, 3])
tensor([1, 2, 1, 1]) # 索引0和3，因为概率为0，因此永不可能被取到
```

#### 2.10 other in-place random sampler

这些采样器是作为tensor类的method存在的。具体可以参考：

- [`torch.Tensor.bernoulli_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli_.html#torch.Tensor.bernoulli_) - in-place version of [`torch.bernoulli()`](https://pytorch.org/docs/stable/generated/torch.bernoulli.html#torch.bernoulli)
- [`torch.Tensor.cauchy_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cauchy_.html#torch.Tensor.cauchy_) - numbers drawn from the Cauchy distribution
- [`torch.Tensor.exponential_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.exponential_.html#torch.Tensor.exponential_) - numbers drawn from the exponential distribution
- [`torch.Tensor.geometric_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.geometric_.html#torch.Tensor.geometric_) - elements drawn from the geometric distribution
- [`torch.Tensor.log_normal_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.log_normal_.html#torch.Tensor.log_normal_) - samples from the log-normal distribution
- [`torch.Tensor.normal_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html#torch.Tensor.normal_) - in-place version of [`torch.normal()`](https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal)
- [`torch.Tensor.random_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html#torch.Tensor.random_) - numbers sampled from the discrete uniform distribution
- [`torch.Tensor.uniform_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html#torch.Tensor.uniform_) - numbers sampled from the continuous uniform distribution



### 3 torch模块下的基础配置函数

#### 3.1 常见标志量函数

torch.is_storage(),torch.is_tensor(),torch.is_complex(),torch.is_floating_point(),

```python
# torch.is_tensor()
a = torch.randn(2,3) 
b = 6
print(a)
print(torch.is_tensor(a))
print(torch.is_tensor(b))
```

```python
# torch.is_storage()
a = torch.rand(2,3) 
b = torch.FloatStorage([1,2,3,4,5,6]) #还有ByteStorage,ShortStorage,IntStorage,LongStorage,DoubleStorage
print(a)
print(a.storage())
print(torch.is_storage(a))
print(torch.is_storage(b))
```

输出为：

```python
tensor([[0.6989, 0.5152, 0.7836],
        [0.8475, 0.9963, 0.1555]])
 0.6989085674285889
 0.5152345299720764
 0.7835797071456909
 0.8474761247634888
 0.9963153004646301
 0.155498206615448
[torch.FloatStorage of size 6]
False
True
```

```python
# torch.is_complex()
a = torch.tensor([2, 4], dtype = torch.float32)
b = torch.tensor([3, 6], dtype = torch.float32)
c = torch.complex(a, b)
print(c) # tensor([2.+3.j, 4.+6.j])
print(torch.is_complex(c)) # True
```

```python
# torch.is_floating_point()
a = torch.tensor([3, 6], dtype = torch.float64)
print(a)# tensor([3., 6.], dtype=torch.float64)
print(torch.is_floating_point(a)) # True
```

对于`torch.is_nonzero()`。如果输入时类型转换后不等于零的单个元素张量，则返回True,即不等于`torch.tensor([0.])`,或`torch.tensor([0])`或`troch.tensor([False])`

```python
# torch.is_nonzero()
print(torch.is_nonzero(torch.tensor([0.])))
print(torch.is_nonzero(torch.tensor([1.5])))
print(torch.is_nonzero(torch.tensor([False])))
print(torch.is_nonzero(torch.tensor([3])))
# output：
False
True
False
True
```

#### 3.2 torch.set_default_dtype(d) 

将默认浮点dtype设置为d，此d一般支持是torch.float32或torch.float64。其他类型虽然不会报错，但是可能无法得到满意结果。

当设置了浮点类型后，complex类型也会随之隐式的改变。

```python
# 如不设置，则输出默认类型，
print(torch.tensor([False]).dtype)
print(torch.tensor([1, 3]).dtype)
print(torch.tensor([1.2, 3]).dtype)
print(torch.tensor([1.2, 3j]).dtype)
# output：
# torch.bool
# torch.int64
# torch.float32
# torch.complex64

# 设置后再次输出
torch.set_default_dtype(torch.float64)
print(torch.tensor([False]).dtype)
print(torch.tensor([1, 3]).dtype)
print(torch.tensor([1.2, 3]).dtype)
print(torch.tensor([1.2, 3j]).dtype) # 隐式改变

# output：
# torch.bool
# torch.int64
# torch.float64
# torch.complex128

```

#### 3.3 torch.get_default_dtype() 

获取当前的默认浮点torch.dtype

```python
print(torch.get_default_dtype()) # torch.float64
torch.set_default_tensor_type(torch.FloatTensor)
print(torch.get_default_dtype()) # torch.float32
```

#### 3.4 torch.set_default_tensor_type(t) 

这个接口和3.2的`torch.set_default_dtype(d)`有点相似，但这个更新，更强大一些，相同的是两者都只设置浮点数默认类型

#### 3.5 torch.numal(input) 

返回输入张量中元素的总数

```python
a = torch.randn(1, 2, 3, 4, 5, 6)
print(torch.numel(a)) # 720
b = torch.zeros(4, 4)
print(torch.numel(b)) # 16
```

#### 3.6 torch.set_printoptions() 

修改打印选项，有时候需要控制输出显示的参数

参数形式：

`torch.set_printoptions`(*precision=None*, *threshold=None*, *edgeitems=None*, *linewidth=None*, *profile=None*, *sci_mode=None*)

Parameters：

- **precision** – Number of digits of precision for floating point output (default = 4).浮点数的显示精度
- **threshold** – Total number of array elements which trigger summarization rather than full repr (default = 1000). 当tensor中的元素过多时，决定在元素数量超过**threshold** 时采用省略号显示。
- **edgeitems** – Number of array items in summary at beginning and end of each dimension (default = 3).当采用省略显示时，边缘保留的元素数量。
- **linewidth** – The number of characters per line for the purpose of inserting line breaks (default = 80). Thresholded matrices will ignore this parameter.每行允许输出的最大字符数量
- **profile** – Sane defaults for pretty printing. Can override with any of the above options. (any one of default, short, full)使用模板的美化输出
- **sci_mode** – Enable (True) or disable (False) scientific notation. If None (default) is specified, the value is defined by torch._tensor_str._Formatter. This value is automatically chosen by the framework是否采用科学计数法

关于**profile** 的用法，这里提供一个规则说明：

```python
if profile is not None:
    if profile == "default":
        PRINT_OPTS.precision = 4
        PRINT_OPTS.threshold = 1000
        PRINT_OPTS.edgeitems = 3
        PRINT_OPTS.linewidth = 80
    elif profile == "short":
        PRINT_OPTS.precision = 2
        PRINT_OPTS.threshold = 1000
        PRINT_OPTS.edgeitems = 2
        PRINT_OPTS.linewidth = 80
    elif profile == "full":
        PRINT_OPTS.precision = 4
        PRINT_OPTS.threshold = inf
        PRINT_OPTS.edgeitems = 3
        PRINT_OPTS.linewidth = 80
```
使用示例：

```python
a = torch.randn(1, 7)
print(a) 
# tensor([[ 1.9613, -0.8069,  0.6356, -0.4459,  0.0860, -0.7173, -1.2878]])
torch.set_printoptions(precision = 6, threshold=6, edgeitems=1, linewidth=20)
print(a) 
# tensor([[-0.326266,
#           ...,
#          -0.461118]])
```



```python
a = torch.randn(1, 3)
print(a) 
# tensor([[ 1.7177,  0.7020, -0.6267,  0.9570, -1.1070]])
torch.set_printoptions(profile='short', sci_mode =True)
print(a) 
# tensor([[1.72e+00, 7.02e-01, -6.27e-01, 9.57e-01, -1.11e+00]])
```









### Reference:

https://pytorch.org/docs/stable/torch.html#tensors

https://pytorch.org/docs/stable/tensors.html#tensor-class-reference

https://blog.csdn.net/MRZHUGH/article/details/112913873

https://blog.csdn.net/Flag_ing/article/details/109129752

https://www.cnblogs.com/CircleWang/p/15658951.html

https://pytorch.org/docs/stable/storage.html?highlight=floatstorage

https://www.cnblogs.com/foghorn/p/15252092.html