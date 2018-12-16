##### ==import torch as t==

torch.zeros()

torch.ones()

torch.save()

torch.load()

torch.sigmoid()

torch.max()

torch.min()

torch.clamp()

torch.sigmoid()

torch.cat()

torch.FloatTensor

torch.cuda.FloatTensor

torch.utils.data.DataLoader()

from troch.utils.data import Dataset

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

##### nn.BatchNorm2d()



##### nn.LeakyReLU()

##### nn.MSELoss()

##### nn.BCELoss()

##### ==import torch.nn.functional as F==



##### ==import torch.optim as optim==

optim.Adam(params, weight_decay=)

optim.Adam(params,weight_decay=, amsgrad=)

optim.RMSprop(params, weight_decay=)

optim.SGD(params, momentum=, weight_decay, nesterov)

optim.lr_scheduler.StepLR()

optimizer.zero_grad()

loss.backward()

optimizer.step()

optimizer.param_groups[]

lr_schedular.step()

##### ==import torchvision==



##### ==import torchvision.transforms as transforms==



##### ==from tensorboardX import SummaryWriter==

config['tensorboard_writer'].add_scalar()

___

##### ==其余杂项==

.new()

torch.from_numpy()

.size(0)

.view().permute().contiguous()

.data

.cuda()

sequeeze()

unsequence()