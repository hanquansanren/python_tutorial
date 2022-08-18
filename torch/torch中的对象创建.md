## 【torch日积月累】之tensor的常用函数集锦

### 1 tensor对象的初始化和复制问题

当定义一个tensor对象时，我们一般会常用`torch.tensor(ndarray or list)`方法，但是该方法仅能够对输入的参数进行深复制，且并不会保留`autograd history`。

> 官方文档原话：torch.tensor constructs a tensor with no autograd history (also known as a “leaf tensor”)

众所周知，python中的对象，普遍存在着赋值和深浅拷贝的问题。在torch框架中，为了应对这一问题，三个常用函数被提出。它们分别是：

#### 1.1 [`torch.as_tensor()`](https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor)



#### 1.2 [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy)





#### 1.3 [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy)

### 2 常用的构造器
#### 2.1  torch.rand
##### 参数形式：

`torch.rand(**size*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*)` → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

##### 功能：依据参数，进行数据类型，所在设备的转换。



##### 使用说明与样例



#### 2.2  其他

##### 参数形式：

`torch.rand(**size*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*)` → [Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)

##### 功能：依据参数，进行数据类型，所在设备的转换。



##### 使用说明与样例



#### Reference:

https://pytorch.org/docs/stable/torch.html#tensors

https://pytorch.org/docs/stable/tensors.html#tensor-class-reference