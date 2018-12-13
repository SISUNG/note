##### ==import torch as t==



##### ==import torch.nn as nn==

##### nn.Linear()

```python
该函数实现了全连接层的功能，即y = wx + b
weight = Parameter(torch.Tensor(out_features, in_features))
bias = Parameter(torch.Tensor(out_features))
```

##### nn.Conv2d()

```python
卷积操作的输出形状计算公式：
output_shape = (in_shape - filter_shape + 2*padding)/stride + 1

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
          padding=0, dilation=1, groups=1, bias=True):
```



##### ==import torch.nn.functional as F==



##### ==import torchvision==



##### ==import torchvision.transforms as transforms==



