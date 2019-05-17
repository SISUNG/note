#### 《Docker教程》

___

相关链接

[菜鸟教程](https://www.runoob.com/docker/docker-tutorial.html)

[Docker官网](http://www.docker.com)

[Github Docker源码](https://github.com/docker/docker)

------

###### Docker教程

0. 阅读本教程前，需要了解的知识

   [Linux教程](https://www.runoob.com/linux/linux-tutorial.html)

1. Docker的应用场景

   - Web应用的自动化打包和发布
   - 自动化测试和持续集成、发布
   - 在服务型环境中部署和调整数据库或其他的应用后台
   - 从头编译或者扩展现有的OpenShift或Cloud Foundry平台来搭建自己的PaaS环境

2. Docker的优点
   - 简化程序
   - 避免选择恐惧症
   - 节省开支

###### Docker架构

0. 理解容器和镜像

   ```
   Docker使用客户端-服务器（C/S）架构模式，使用远程API来管理和创建Docker容器；Docker容器通过Docker镜像来创建；容器与镜像的关系类似于面向对象编程中的对象和类
   ```

1. 概念简介

   | 名词                    | 名词解释                                                     |
   | ----------------------- | ------------------------------------------------------------ |
   | Docker镜像（Image）     | Docker镜像是用于创建Docker容器的模板                         |
   | Docker容器（Container） | 容器是独立运行的一个或一组应用                               |
   | Docker客户端（Client）  | Docker 客户端通过命令行或者其他工具使用 Docker API (<https://docs.docker.com/reference/api/docker_remote_api>) 与 Docker 的守护进程通信。 |
   | Docker 主机(Host)       | 一个物理或者虚拟的机器用于执行 Docker 守护进程和容器。       |
   | Docker 仓库(Registry)   | Docker 仓库用来保存镜像，可以理解为代码控制中的代码仓库。Docker Hub([https://hub.docker.com](https://hub.docker.com/)) 提供了庞大的镜像集合供使用。 |
   | Docker Machine          | Docker Machine是一个简化Docker安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装Docker，比如VirtualBox、 Digital Ocean、Microsoft Azure。 |

##### Docker安装

###### Ubuntu Docker安装

0. 前提条件

   ```
   
   ```

##### CentOS Docker安装

##### Windows Docker安装

##### MacOS Docker安装

##### Docker使用

###### Docker Hello World

###### Docker容器使用

###### Docker镜像使用

0. 创建镜像
   - 从已经创建的容器中更新镜像，并且提交这个镜像
   - 使用Dockerfile指令来创建一个新的镜像

1. 更新镜像
2. 构建镜像
3. 设置镜像标签

###### Docker容器连接

0. 网络端口映射
1. 容器命名

##### Docker实例

###### Docker安装Nginx

###### Docker安装PHP

###### Docker安装MySQL

###### Docker安装Tomcat

###### Docker安装Python

###### Docker安装Redis

###### Docker安装MongoDB

###### Docker安装Apache

##### Docker参考手册

###### Docker命令大全

###### Docker资源汇总

___

Docker常用命令总结

```
docker run ubuntu /bin/echo "Hello world"
Docker以ubuntu镜像创建一个新容器，然后在容器里执行bin/echo "Hello world"，然后输出结果

docker run -i -t ubuntu /bin/bash
运行交互式的容器，让docker运行的容器实现“对话”的能力
cat /proc/version查看当前系统的版本信息
ls 查看当前目录下的文件列表

运行exit命令或者使用CTRL+D来退出容器

docker run -d ubuntu /bin/sh -c "while true;do echo hello world;sleep 1;done"
创建一个以进程方式运行的容器，得到长字符串容器ID

docker ps
查看是否有容器在运行
CONTAINER ID:容器ID
NAMES:自动分配的容器名称

docker logs 91225572afc8或者docker logs happy_dubinsky
在容器内使用docker logs命令，查看容器内的标准输出

docker stop 91225572afc8或者docker stop happy_dubinsky
停止容器

docker
查看Docker客户端的所有命令选项

docker command --help
更深入地了解指定的Docker命令使用方法

docker pull training/webapp  # 载入镜像
docker run -d -P training/webapp python app.py（大写P）
-d让容器在后台运行
-P将容器内部使用的网络端口映射到我们使用的主机上
在Docker容器中运行一个Python Flask应用来运行一个web应用
可以通过浏览器localhost:5636访问WEB应用

docker run -d -p 5000:5000 training/webapp python app.py（小写p）
通过-p来设置不一样的端口

docker port bf08b7f2cd89 或 docker port wizardly_chandrasekhar
使用 docker port 可以查看指定 （ID 或者名字）容器的某个确定端口映射到宿主机的端口号

docker logs -f [ID或者名字]
像使用 tail -f 一样来输出容器内部的标准输出

docker top wizardly_chandrasekhar
查看容器内部运行的进程

docker inspect wizardly_chandrasekhar
查看 Docker 的底层信息。它会返回一个 JSON 文件记录着 Docker 容器的配置和状态信息

docker stop wizardly_chandrasekhar
停止WEB应用容器

docker start wizardly_chandrasekhar
重启WEB应用容器

docker ps -l
查询最后一次创建的容器

docker restart wizardly_chandrasekhar
重启WEB应用容器

docker rm wizardly_chandrasekhar
删除WEB应用容器，删除容器时，容器必须是停止状态，否则会报如下错误

docker search httpd
查找镜像

docker pull httpd
拖取镜像

runoob@runoob:~$ docker run -t -i ubuntu:15.10 /bin/bash
root@e218edb10161:/# 
在运行的容器内使用 apt-get update 命令进行更新。
在完成操作之后，输入 exit命令来退出这个容器。
此时ID为e218edb10161的容器，是按我们的需求更改的容器。我们可以通过命令 docker commit来提交容器副本。
runoob@runoob:~$ docker commit -m="has update" -a="runoob" e218edb10161 runoob/ubuntu:v2
sha256:70bf1840fd7c0d2d8ef0a42a817eb29f854c1af8f7c59fc03ac7bdee9545aff8
-m:提交的描述信息
-a:指定镜像作者
e218edb10161：容器ID
runoob/ubuntu:v2:指定要创建的目标镜像名


docker images [OPTIONS] [REPOSITORY[:TAG]]
列出本地镜像
-a :列出本地所有的镜像（含中间映像层，默认情况下，过滤掉中间映像层）；
--digests :显示镜像的摘要信息；
-f :显示满足条件的镜像；
--format :指定返回值的模板文件；
--no-trunc :显示完整的镜像信息；
-q :只显示镜像ID

```

