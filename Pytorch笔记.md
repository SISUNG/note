##### Win10 64bit PyTorch-cpu1.0安装

```python
#for 0.4.0 and later更换为清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
#for CPU only packages
conda install pytorch-cpu
```

##### PyTorch常用模块整理

| 主模块           | 子模块     |                                                              |
| ---------------- | ---------- | ------------------------------------------------------------ |
| torch.utils.data | Dataset    |                                                              |
|                  | DataLoader |                                                              |
|                  |            |                                                              |
|                  |            |                                                              |
| torchvision      | models     | from .alexnet import * <br />from .resnet import * <br />from .vgg import * <br />from .squeezenet import * <br />from .inception import * <br />from .densenet import * |
|                  | datasets   | from .lsun import LSUN, LSUNClass <br />from .folder import ImageFolder, DatasetFolder <br />from .coco import CocoCaptions, CocoDetection <br />from .cifar import CIFAR10, CIFAR100 <br />from .stl10 import STL10 <br />from .mnist import MNIST, EMNIST, FashionMNIST <br />from .svhn import SVHN <br />from .phototour import PhotoTour <br />from .fakedata import FakeData <br />from .semeion import SEMEION <br />from .omniglot import Omniglot |
|                  | transforms |                                                              |
|                  | utils      | make_grid<br />save_image                                    |
|                  |            |                                                              |
|                  |            |                                                              |
|                  |            |                                                              |
|                  |            |                                                              |
|                  |            |                                                              |



___

##### PyTorch模型训练实用教程

##### 前言：

- 推出于2017年1月
- 追求者：众多==研究人员==和工程师
- 优点：采用Python语言、动态图机制、网络构建灵活以及拥有强大的社群等
- 机器学习模型开发三大部分：数据、模型和损失函数及优化器
- 当模型训练遇到问题时，需要通过可视化工具对数据、模型、损失等内容进行观察，分析并定位问题出处
- 工程应用开发所遇到的问题：数据预处理、数据增强、模型定义、权值初始化、模型Finetune、学习率调整策略、损失函数选取、优化器选取、可视化等等
- 主要内容：在PyTorch中训练一个模型所可能涉及到的方法及函数，并且对PyTorch提供的数据增强方法（22个）、权值初始化方法（10个）、损失函数（17个）、优化器（6个）及tensorboardX的方法（13个）



##### 第一章 数据

##### 1.1 Cifar10转png

> 第一步：下载cifar-10-python.tar.gz并且解压
>
> 第二步：运行1_1_cifar10_to_png.py

##### 1.2 训练集、验证集和测试集的划分

> 训练集（train set）：验证集（valid/dev set)：测试集（test set）= 8:1:1
>
> 运行Code/1_data_prepare/1_2_split_dataset.py

##### 1.3 让PyTorch能读懂你的数据集

- 想让PyTorch能读取我们自己的数据，首先要了解PyTorch读取图片的机制和流程，然后按流程编写流程

##### Dataset类

> Dataset类作为所有的datasets的基类存在，所有的datasets都需要继承它，类似于c++中的==虚基类==

```python
class Dataset(object):
    def __getitem__(self,index):
        raise NotImplementedError
  	#getitem接收一个index，然后返回图片数据和标签，这个index通常指的是一个list的index，这	个list的每个元素就包含了图片数据的路径和标签信息
    def __len__(self):
        raise NotImplementedError
    def __add__(self,other):
        return COncatDataset([self,other])
```

> 读取自己数据的基本流程：
>
> 制作存储了图片的路径和标签信息的txt
>
> 将这些信息转化为list，该list每一个元素对应一个样本
>
> 通过getitem函数，读取数据和标签，并返回数据和标签

##### 1.制作图片数据的索引

> 读取图片路径，标签，保存到txt文件中，注意格式就好。特别注意的是，txt中的路径，是以训练时的那个py文件所在的目录为工作目录，所以这里需要提前算好相对路径

##### 2.构建Dataset子类

```python
from PIL import Image
from torch.utils.data import Dataset
```

> 初始化还会初始化transform，transform是一个Compose类型，里面有一个list，list中就会定义了各种对图像进行处理的操作，可以设置减均值，除标准差，随机裁剪，旋转，翻转，仿射变换等操作
>
> 当Mydataset构建好后，剩下的操作就交给DataLoader,在DataLoader中，会触发Mydataset中的getitem函数读取一张图片的数据和标签，并拼接成一个batch返回，作为模型真正的输入

##### ==1.4 图片从硬盘到模型==



##### 1.5数据增强与数据标准化

> 数据中心化（仅减均值），数据标准化（减均值，再除以标准差），随机裁剪，旋转一定角度，镜像等一系列操作
>
> transforms.py
>
> transforms.Normalize()
>
> transforms.Compose()
>
> transforms.Resize()
>
> transforms.RandomCrop()
>
> transform.ToTensor()

##### 1.随机裁剪

```python
transforms.RandomCrop(32,padding=4)	#在裁剪之前先对图片的上下左右均填充4个pixel，值为0，即变成一个36*36的数据，然后再随机进行32*32的裁剪
```

##### 2.ToTensor

```
transforms.ToTensor()
在这里会对数据进行transpose，原来是h*w*c，会经过img=img.transpose(0,1).transpose(0,2).contiguous()，变成c*h*w再除以255，使得像素值归一化
```

##### 3.数据标准化（减均值，除以标准差）

> 至此，数据预处理完毕，最后转换成Variable类型，就是输入网络模型的数据了

##### 1.6 transforms的二十二个方法

##### 1.随机裁剪：transforms.RandomCrop

```python
class torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False,fill=0,padding_mode='constant')
```

##### 2.中心裁剪：transforms.CenterCrop

```python
class torchvision.transforms.CenterCrop(size)
功能：依据给定的size从中心裁剪
```

##### 3.随机长宽比裁剪：transforms.RandomResizedCrop

```python
class torchvision.transforms.RandomResizedCrop(size,scale=(0.08,1.0),
                                ratio=(0.75,1.33333),interpolation=2)
功能：随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size
```

##### 4.上下左右中心裁剪：transforms.FiveCrop

```python
class torchvision.transforms.FiveCrop(size)
功能：对图片进行上下左右以及中心裁剪，获得5张图片，返回一个4D-tensor
```

##### 5.上下左右中心裁剪后翻转：transforms.TenCrop

```python
class torchvision.transforms.TenCrop(size,vertical_flip=False)
功能：对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得10张图片，返回一个4D-tensor
```

##### 6.依概率p水平翻转transforms.RandomHorizontalFlip

```python
class torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

##### 7.依概率p垂直翻转transforms.RandomVerticalFlip

```python
class torchvision.transforms.RandomVerticalFlip(p=0.5)
```

##### 8.随机旋转：transforms.RandomRotation

```python
class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
功能：依degrees随机旋转一定角度
```

##### 9.resize：transforms.Resize

```python
class torchvision.transform.Resize(size, interpolation=2)
功能：重置图像分辨率
```

##### 10.标准化：transforms.Normalize

```python
class torchvision.transforms.Normalize(mean,std)
功能：对数据按通道进行标准化，即先减均值，再除以标准差，注意是h*w*c
```

##### 11.转为tensor：transforms.ToTensor

```python
class torchvision.transforms.ToTensor
功能：将PIL Image或者ndarray转换为tensor，并且归一化
```

##### 12.填充：transforms.Pad

```python
class torchvision.transforms.Pad(padding,fill=0,padding_mode='constant')
功能：对图像进行填充
```

##### 13.修改亮度、对比度和饱和度：transforms.ColorJitter

```python
class torchvision.transforms.ColorJitter(brightness=0,
                                         contrast=0, saturation=0, hue=0)
```

##### 14.转灰度图：transforms.Grayscale

```python
class torchvision.transforms.Grayscale(num_output_channels=1)
功能：将图片转换为灰度图
```

##### 15.线性变换：transforms.LinearTransformation()

```python
class torchvison.transforms.LinearTransformation(transformation_matrix)
功能：对矩阵做线性变化，可用作白化处理
```

##### 16.仿射变换：transforms.RandomAffine

```python
class torchvision.transforms.RandomAffine(degrees, translate=None,
                                         scale=None, shear=None,
                                         resample=False,
                                         fillcolor=0)
功能：仿射变换
```

##### 17.依概率p转为灰度图：transforms.RandomGrayscale

```python
class torchvision.transforms.RandomGrayscale(p=0.1)
功能：依概率p将图片转换为灰度图
```

##### 18.将数据转换为PILImage：transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(mode=None)
功能：将tensor或者ndarray的数据转换为PIL Image类型数据
mode-为None时，为1通道，mode=3通道默认转换为RGB，4通道默认转换为RGBA
```

##### 19.transforms.Lambda

```python
Apply a user-defined lambda as a transform.
```

##### 20.transforms.RandomChoice(transforms)

```python
功能：从给定的一系列transforms中选一个进行操作，randomly picked from a list
```

##### 21.transforms.RandomApply(transforms,p=0.5)

```python
功能：给一个transforms加上概率，以一定的概率执行该操作
```

##### 22.transforms.RandomOrder

```python
功能：将transforms中的操作顺序随机打乱
```







##### 第二章 模型

##### 2.1 模型的搭建

##### 2.1.1 模型定义的三要

>首先，必须继承nn.Module这个类，要让PyTorch知道这个类是一个Module
>
>其次，在__ init __(self)中设置好需要的组件，如conv、pooling、Linear、BatchNorm等
>
>最后，在forward（self，x）中用定义好的组件进行组装，就像搭积木，把网络结构搭建起来，这样一个模型就定义好了
>
>至此，一个模型定义完毕，接着就可以在后面进行使用。例如，实例化一个模型net=Net(),然后把输入inputs扔进去，outputs=net(inputs),就可以得到输出outputs

##### 2.1.2 模型定义多说两句

##### 2.1.3 nn.Sequential()

> torch.nn.Sequtntial其实就是Sequential容器，该容器将一系列操作按先后顺序给包起来，方便重复使用。
>
> 模型的定义就是先继承，再构建组件，最后组装。其中基本组件可从torch.nn获取，或者从torch.nn.functional中获取，同时为了方便重复使用组件，可以使用Sequential容器将一系列组件包起来，最后在forward()函数中将这些组件组装成你的模型

##### 2.2 权值初始化的十种方法

##### 2.2.1 权值初始化流程

> 第一步，先设定什么层用什么初始化方法，初始化方法在torch.nn.init中给出
>
> 第二步，实例化一个模型之后，执行该函数，即可完成初始化

```python
self.modules()源码在torch/nn/modules/module.py中
def modules(self):
    for name, module in self.name_modules():
        yield module
#功能是returns an iterator over all modules in the network能依次返回模型中的各层
```

##### 2.2.2 常用初始化方法

##### 1.Xavier均匀分布

> torch.nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
>
> xavier初始化方法中服从均匀分布U(-a,a)，分布的参数a=gain*sqrt(6/fan_in+fan_out),
>
> 上述初始化方法，也称为Glorot initialization

##### 2.Xavier正态分布

> torch.nn.init.xavier_normal_(tensor,gain=1)
>
> xavier初始化方法中服从正态分布，mean=0, std=gain*sqrt(2/fan_in+fan_out)

##### 3.kaiming均匀分布

> torch.nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
>
> mode可选为fan_in或fan_out，fan_in使正向传播时，方差一致；fan_out使反向传播时，方差一致

##### 4.kaiming正态分布

> torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')

##### 5.均匀分布初始化

>torch.nn.init.uniform_(tensor, a=0, b=1)
>
>使值服从均匀分布U(a,b)

##### 6.正态分布初始化

> torch.nn.init.normal_(tensor, mean=0, std=1)
>
> 使值服从正态分布N(mean,std), 默认值为0,1

##### 7.常数初始化

> torch.nn.init.constant_(tensor, val)
>
> 使值为常数val	nn.init.constant_(w, 0.3)

##### 8.单位矩阵初始化

> torch.nn.init.eye_(tensor)
>
> 将二维tensor初始化为单位矩阵（the identity matrix)

##### 9.正交初始化

> torch.nn.init.orthogonal_(tensor, gain=1)

##### 10.稀疏初始化

> torch.nn.init.sparse_(tensor, sparsity, std=0.01)
>
> 从正态分布N(0, std)中进行稀疏化，使每一个column中有一部分为0，sparsity为每一个column稀疏的系数

##### 11.计算增益

> torch.nn.init.calculate_gain(nonlinearity, param=None)

##### 权值初始化杂谈

>1.即使不进行初始化，模型的权值也不为空，而是有值的。其实，在创建网络实例的过程中，一旦调用nn.Conv2d的时候就会对权值进行初始化。Conv2d是继承_ConvNd，初始化赋值是在_ConvNd当中的self.weight进行的。
>
>2.按需定义初始化方法
>
>```python
>if isinstance(m, nn,Conv2d):
>    n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
>    m.weight.data.normal_(0, math.sqrt(2./n))
>```

##### 2.3模型Finetune

- 通常采用一个已经训练好的模型的权值参数作为我们模型的初始化参数，也称之为Finetune，更宽泛地称之为迁移学习。迁移学习中的Finetune技术，本质上就是让我们新构建的模型，拥有一个较好的权值初始值
- FInetune权值初始化三部曲：第一步，保存模型，拥有一个预训练模型；第二步，加载模型，把预训练模型的权值取出来；第三步，初始化，将对应的权值放到新模型中

##### 一、Finetune之权值初始化

- 官方文档中介绍了两种保存模型的方法，一种是保存整个模型，另一种是仅保存模型参数（官方推荐用这种方法）

##### 第一步：保存模型参数

```python
net = Net()
torch.save(net.state_dict(), 'net_params.pkl')
```

##### 第二步：加载模型

```python
pretrained_dict=torch.load('net_params.pkl')
```

##### 第三步：初始化

```python
首先创建新模型，并且获取新模型的参数字典net_state_dict：
net=Net()#创建net
net_state_dict=net.state_dict()#获取已创建net的state_dict
接着将pretrained_dict里不属于net_state_dict的键剔除掉：
pretrained_dict_l={k:v for k,v in pretrained_dict.items() 
                   if k in net_state_dict}
然后，用预训练模型的参数字典对新模型的参数字典net_state_dict进行更新：
net_state_dict.update(pretrained_dict_l)
最后，将更新了参数的字典放回到网络中：
net.load_state_dict(net_state_dict)
这样利用预训练模型参数对新模型的权值进行初始化过程就做完了
```

##### 二、不同层设置不同的学习率

- 在利用pre-trained model的参数做初始化之后，我们可能想让fc层更新相对快一些，而希望前面的权值更新小一些，这就可以通过为不同的层设置不同的学习率来达到此目的
- 为不同层设置不同的学习率，主要通过优化器对多个参数组设置不同的参数。所以，只需要将原始的参数组，划分成两个，甚至更多的参数组，然后分别进行设置学习率





##### 第三章 损失函数与优化器

- 本章的主要目的是选择合适的损失函数，并且采用合适的优化器进行优化（训练）模型。
- 将介绍PyTorch中的十七个损失函数，十个优化器和六个学习率调整方法

##### 3.1 PyTorch的十七个损失函数

##### 17.TripletMarginLoss

```

```



##### 3.2优化器基类：Optimizer

- PyTorch中所有的优化器均是Optimizer的子类，Optimizer中定义了一些常用的方法，有zero_grad()、step(closure)、state_dict()、load_state_dict(state_dict)和add_param_group(param_group)，本节将会一一介绍

##### 3.2.1 参数组(param_groups)的概念

>optimizer对参数的管理是基于组的概念，可以为每一组参数配置特定的lr，momentum，weight_decay等等。参数组在optimizer中表现为一个list（self.param_groups),其中每个元素是dict，表示一个参数及其相应配置，在dict中包含'params'	'weight_decay'	'lr'	'momentum'等字段

##### 3.2.2 zero_grad()

> 功能：将梯度清零
>
> 由于PyTorch不会自动清零，所以在每一次更新前会进行此操作

##### 3.2.3 state_dict()

> 功能：获取模型当前的参数，以一个有序字典形式返回。这个有序字典中，key 是各层参数名，value 就是参数。

##### 3.2.4 load_state_dict(state_dict)

> 功能：将 state_dict 中的参数加载到当前网络，常用于 finetune。

##### 3.2.5 add_param_group()

> 功能：给 optimizer 管理的参数组中增加一组参数，可为该组参数定制 lr,momentum,weight_decay 等，在 finetune 中常用。

##### 3.2.6 step(closure)

>功能：执行一步权值更新, 其中可传入参数 closure（一个闭包）。如，当采用 LBFGS
>优化方法时，需要多次计算，因此需要传入一个闭包去允许它们重新计算 loss

##### 3.3PyTorch的十个优化器

##### 1.torch.optim.SGD

##### 2.torch.optim.ASGD

##### 3.torch.optim.Rprop

##### 4.torch.optim.Adagrad

##### 5.torch.optim.Adadelta

##### 6.torch.optim.RMSprop

##### 7.torch.optim.Adam(AMSGrad)

##### 8.torch.optim.Adamax

##### 9.torch.optim.SparseAdam

##### 10.torch.optim.LBFGS

##### 3.4 PyTorch的六个学习率调整方法

##### 1.lr_scheduler.StepLR

```python
class torch.optim.lr_scheduler.StepLR(optimizer,step_size,gamma=0.1,
                                     last_epoch=-1)
功能：等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。间隔单位是step。需要注意的是，step通常是指epoch，不要弄成iteration
```

##### 2.lr_scheduler_MultiStepLR

```python
class torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=0.1,
                                           last_epoch=-1) 
功能：按设定的间隔调整学习率。这个方法适合后期调试使用，观察loss曲线，为每个实验定制学习率调整时机
```

##### 3.lr_scheduler.ExponentialLR

```python
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, 
                                             last_epoch=-1)
功能：按指数衰减调整学习率
```

##### 4.lr_scheduler.CosineAnnealingLR

##### 5.lr_scheduler.ReduceLROnPlateau

##### 6.lr_scheduler.LambdaLR

##### 学习率调整小结

> 可分为三大类，分别是：有序调整；自适应调整；自定义调整



##### 第四章 监控模型-可视化

- 本章将介绍如何在PyTorch中使用TensorBoardX对神经网络进行统计可视化，如loss曲线、accuracy曲线、卷积核可视化、权值直方图及多分位数折线图、特征图可视化、梯度直方图及多分位数折线图及混淆矩阵图等

##### 4.1 TensorBoardX

##### 1.构建Logger类

> Logger类中包含了tf.summary.FileWriter，目前只有三种操作，分别是scalar_summary()、image_summary()、histo_summary()

##### 2.借助TensorBoardX包

> TensorBoardX包的功能就比较全，支持除tensorboard beholder之外的所有tensorboard的记录类型

##### 安装时小插曲：

> pip uninstall tensorboardX
>
> pip install tensorboard安装成功
>
> tensorboardX最早叫tensorboard，但此名易引起混淆，之后改成tensorboardX

##### 代码实现

>先运行1_tensorboardX_demo.py，并且打开terminal，进入相应的虚拟环境，进入到/Result/文件夹，执行：tensorboard --logdir=runs
>
>然后到浏览器中打开：localhost：6006

##### tensorboardX的函数

##### 1.add_scalar()

```python
add_scalar(tag,scalar_value,global_step=None,walltime=None)
功能：在一个图表中记录一个标量的变化，常用于Loss和Accuracy曲线的记录
```

##### 2.add_scalars()

```python
add_scalars(main_tag,tag_scalar_dict,global_step=None,walltime=None)
功能：在一个图表中记录多个标量的变化，常用于对比，如trainloss和validloss的比较等
```

##### 3.add_histogram()

```python
add_histogram(tag,values,global_step=None,bins='tensorflow',walltime=None)
功能：绘制直方图和多分位数折线图，常用于监测权值及梯度的分布变化情况，便于诊断网络更新方向是否正确
for name, param in resnet18.named_parameters():
    writer.add_histogram(name,param.clone().cpu().data.numpy(),n_iter)
可以得到以下两种图分别在HISTOGRAMS和DISTRIBUTIONS里面
```

##### 4.add_image()

```
add_image(tag,img_tensor,global_step=None,walltime=None)
功能：绘制图片，可用于检查模型的输入，监测feature map的变化，或是观察weight
shape = [C, H, W]
```

通常会借助torchvision.utils.make_grid()将一组图片绘制到一个窗口

##### 补充torchvision.utils.make_grid()

```python
torchvision.utils.make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0)
功能：将一组图片拼接成一张图片，便于可视化
import torchvision.utils as vutils
dummy_img = torch.rand(32,3,64,64)
if n_iter % 10 == 0:
    x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
    writer.add_image('Image', x, n_iter)	#x.size=(3,266,530)
```

##### 5.add_graph()

```python
add_graph(model,input_to_model=None,verbose=False, **kwargs)
功能：绘制网络结构拓扑图
import torchvision.models as models
resnet18 = models.resnet18(False)
dummy_input = torch.rand(6,3,224,224)
writer.add_graph(resnet18, dummy_input)
```

##### 6.add_embedding()

```python
add_embedding(mat,metadata=None,label_img=None,global_step=None,tag='default',metadata_header=None)
功能：在三维空间或二维空间展示数据分布，可选T-SNE、PCA、和CUSTOM方法
```

##### 7.add_text()

```python
add_text(tag, text_string,global_step=None,walltime=None)
功能：记录文字
```

##### 8.add_video()

```python
add_video(tag,vid_tensor,global_step=None,fps=4,walltime=None)
功能：记录video	
```

##### 9.add_figure()

```python
add_figure(tag,figure,global_step=None,close=True,walltime=None)
功能：添加matplotlib图片到图像中         
```

##### 10.add_image_with_boxes()

```python
add_image_with_boxes(tag,img_tensor,box_tensor,global_step=None,
                     walltime=None, **kwargs)
功能：图像中绘制Box，目标检测中会用到
```

##### 11.add_pr_curve()

```python
add_pr_curve(tag,labels,predictions,global_step=None,num_thresholds=127,
            weights=None, walltime=None)
功能：绘制PR曲线
```

##### 12.add_pr_curve_raw()

```python
add_pr_curve_raw(tag,...)
功能：从原始数据上绘制PR曲线
```

##### 13.export_scalars_to_json()

```python
export_scalars_to_json(path)
功能：将scalars信息保存到json文件，便于后期使用
```

##### 4.2 卷积核可视化

- 可视化原理很简单，对单个卷积核进行“归一化”至0-255，然后将其展现出来即可，这一系列操作可以借助TensorboardX的add_image来实现
- 具体卷积核过程可以观察下图，输入通道数为4，输出通道数为2，卷积层卷积核有8个2*2的卷积核

##### 代码实现：

```python
执行：tensorboard --logdir=visual_weights
进入浏览器，打开网页：http://localhost:6006/
```

##### 4.3 特征图可视化

##### 基本思路：

```python
1.获取图片，将其转换成模型输入前的数据格式，即一系列transform
2.获取模型各层操作，手动执行每一层操作，拿到所需的feature maps
3.借助tensorboardX进行绘制
```

##### Tips:

```python
此处获取模型各层操作是__ init __()中定义的操作，然而模型真实运行采用的是forward(),所以需要人工对比两者差异。本例的差异是，__ init __()中缺少激活函数relu
```

##### 4.4 梯度及权值分布可视化

- 在网络训练过程中，我们常常会遇到梯度消失、梯度爆炸等问题，我们可以通过记录每个epoch的梯度的值来监测梯度的情况，还可以记录权值，分析权值更新的方向是否符合规律

##### 可视化分析：

```python
1.权值weights的监控
若权值太大容易导致过拟合，因为模型的输出值会被该特征所主导，从而引起过拟合现象，这个可以通过权值衰减(weight_decay)来缓解
2.偏置bias的监控
通常会监控输出层的bias的大小，若有特别大或者特别小的bias，那么这一类别的召回率可能会很低，可以通过观察输出层的bias来诊断是否在这一环节出问题
3.梯度的监控
倘若前面几层的梯度非常小，那么就是梯度流通不畅导致的，可以考虑残差结构或者辅助损失层等trick来解决梯度消失
```

##### 4.5混淆矩阵及其可视化

- 在分类任务中，个人十分喜欢混淆矩阵，通过混淆矩阵可以看出模型的偏好，而且对每一个类别的分类情况都了如指掌，为模型的优化提供很大帮助

##### 1.混淆矩阵概念

> 混淆矩阵（confusion matrix）常用来观察分类结果，其中一个N*N的方阵，N表示类别数。混淆矩阵的行表示真实类别，列表示预测类别。
>
> 召回率（Recall）
>
> 精确率（Precision）
>
> 准确率（Accuracy）

##### 2.混淆矩阵的统计

> 第一步：创建混淆矩阵，获取类别数，创建N*N的零矩阵
>
> 第二步：获取真实标签和预测标签，labels为真实标签，通常为一个batch的标签；predicted为预测类别，与labels同长度
>
> 第三步：依据标签为混淆矩阵计数

##### 3.混淆矩阵可视化



##### 结束语

> 内容到此结束，希望本教程能给大家带来帮助！

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

torch.nn

torch.nn.functional

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

nn.ModuleList()

##### nn.LeakyReLU()

##### ==Loss function总结==

```python
nn.L1Loss()

nn.SmoothL1Loss()

nn.MSELoss() 

nn.BCELoss()
nn.BCEWithLogitsLoss()
nn.CrossEntropyLoss()
nn.NLLLoss() #用于多分类的负对数似然损失函数
nn.NLLLoss2d()
nn.KLDivLoss() #KL散度
nn.MarginRankingLoss() #评价相似度的损失
nn.MultiMarginLoss() #多分类的Hinge损失
nn.SoftMarginLoss() #多标签二分类损失
nn.MultiLabelSoftMarginLoss() #上面的多分类版本
nn.CosineEmbeddingLoss() #余弦相似度的损失
nn.HingeEmbeddingLoss()
nn.TripleMarginLoss() #我也不知道这是什么鬼东西？？？
```



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

model.load_state_dict(torch.load(pretrained))

.repeat()