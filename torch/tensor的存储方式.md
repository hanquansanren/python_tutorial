## 详谈tensor和numpy对多维数组存储方式，以及高级索引机制

1 When accessing the contents of a tensor via indexing, PyTorch follows Numpy behaviors that **basic indexing returns views**, while **advanced indexing returns a copy**.
**2 Assignment via either basic or advanced indexing is in-place**.
See more examples in[Numpy indexing documentation](https://link.zhihu.com/?target=https%3A//docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

### 1多维数组的存储及其常见误区

tensor对象中对于多维数组的存储，和list对象（采用多层引用，实质上就是指向关系）迥然不同。tensor对象内部是一个wrapper（包装器），他包含一个头信息区（`Tensor`）和一个存储区 （`Storage`）。如下图所示：

![](D:\E\py_project\usage_numpy\python_tutorial\torch\figures\20201017124805973.png)

比如，我们定义并初始化一个tensor，tensor名为A，A的形状size、步长stride、数据的`index(),slice(),cat(),view()`等操作信息等信息都存储在头信息区，而A所存储的真实数据则存储在存储区。另外，如果我们对A进行截取、转置或修改等操作后赋值给B，则B的数据共享A的存储区，存储区的数据数量没变，变化的只是B的头信息区对数据的索引方式。当有需要访问其存储区时，则需要调用`self.storage()`才能获得。

例如，当我们使用切片操作时，torch实际上是调用了tensor的内置方法 `self.__getitem__()`来创建新的wrapper对象，实际访问的头信息区并不影响内存中真实存储的数据。而list对象在切片时，则是调用了`self.index()`方法，直接访问了真实内存存储的地址，通过截取的指定切片的真实存储内存地址作为一个新的对象。新对象（切片后）和旧对象的重合部分，在底层依然占用着相同的内存地址。

一个常见的误区：对于普通的python对象，我们常用`id()`去获取其真实的内存地址（基于CPython解释器）。当使用`id()`获取切片对象的内存地址时，`id(a[0])`无法访问到真实的存储区；而`id(a[0].storage())`则会返回一个新的storage对象的地址。两者均不能访问到真实存储的切片所在位置，**由此可见，在numpy以及torch等多位矩阵处理框架下，使用`id()`去试图获得内存地址是非常错误的。**为了解决这一问题，torch提供了一个来自于C++底层的方法，该方法可以返回真实的内存地址，即为`a[0].storage().data_ptr()`。而在numpy中，一般采用`a.__array_interface__['data'][0]`。

#### 1.1 例子1

下面是一段示例代码：

```python
# 切片时的神奇现象
a = torch.tensor([0,1,2,3])
print(id(a[0]))
print(id(a[0].storage()))
print(a[0].storage().data_ptr())
print("###########################")

print(id(a[0]))
print(id(a[0].storage()))
print(a[0].storage().data_ptr())
print("###########################")

print(id(a[0]))
print(id(a[0].storage()))
print(a[0].storage().data_ptr())

```

输出为：

```python
1472003733760
1472006607936
1471976496384
###########################
1472043755296
1472006586624
1471976496384
###########################
1472043753616
1472006673728
1471976496384
```

可以看出，每次调用`id(a[0])`和`id(a[0].storage())`，id均会改变，这说明他们并未真正指向数据真实存储的内存地址。

#### 1.2 例子2

另一个来自[知乎链接](https://www.zhihu.com/question/433799503)提问的对照实验，我将其作为一个素材：

```python
a = torch.tensor([[1,2],[3,4]])
print(id(a), id(a[0]), id(a[1]))
a[1:] = torch.tensor([0, 0])
print(id(a), id(a[0]), id(a[1])) # 2257492943160 2257492943320 2257492943320 ←①
a0 = a[0]
print(id(a0)) # 2257492943320
a1 = a[1]
print(id(a1)) # 2257492943400 ←②
```

再使用普通的list对象，重复相同的工作：

```python
a = [[1,2],[3,4]]
print(id(a), id(a[0]), id(a[1])) # 2559517439872 2559383846848 2559517746368
a[1:] = [[0, 0]] 
print(a) # [[1, 2], [0, 0]]
print(id(a), id(a[0]), id(a[1])) # 2559517439872 2559383846848 2559517718720 ←①

a0 = a[0]
print(id(a0)) # 2559383846848
a1 = a[1]
print(id(a1)) # 2559517718720 ←②
```

注意看两个数字①和②。也可以证明上述的作者的观点是正确的。

### 2 高级索引机制







### reference

https://github.com/leisurelicht/wtfpython-cn

https://discuss.pytorch.org/t/id-function-of-tensor-changes-all-the-time/97570/3

https://discuss.pytorch.org/t/why-is-id-tensors-storage-different-every-time/100478

https://blog.csdn.net/wohu1104/article/details/107419486

https://www.zhihu.com/question/433799503