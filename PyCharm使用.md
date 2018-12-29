##### 1.常用快捷键总结

| 快捷键   | 功能         | 快捷键       | 功能     |
| -------- | ------------ | ------------ | -------- |
| Alt+F1   | 查找当前文件 | Ctrl+Shift++ | 展开代码 |
| Ctrl+Tab | 切换窗口     | Ctrl+Shift+- | 收缩代码 |
| Ctrl+X   | 剪贴         |              |          |
| Ctrl+B   | 看声明       |              |          |
| Ctrl+N   | 查找类名     |              |          |
| Shift+F9 | 调试         |              |          |

##### 2.快速查看库源码

Ctrl+B或者Ctrl+鼠标左键

##### 3.快速换行

Shift+Enter

##### 4.重新设置万能提示键

单击空格键

##### 5.历史粘贴板

Ctrl+Shift+V

##### 6.快速运行代码

Ctrl+Shift+F10

##### 7.切分屏幕

在Settings里可以设置

##### 8.快速展开和合并函数

快速展开Ctrl+Shift++

快速合并Ctrl+Shift+-

##### 9.快速注释（或者快速取消注释）

Ctrl+/

##### 10.超级搜索

双击Shift

搜索目录的技巧是在关键字前面加上斜杠/

##### 11.打开最近访问过的文件

Ctrl+E

##### 12.分割窗口

纵向分割Alt+V

横向分割Alt+H

##### 13.整段缩进（或取消整段缩进）

选中内容+Tab

选中内容+Shift+Tab

##### 14.快速定位到代码块的开始结束

快速定位到开始Ctrl+[

快速定位到结束Ctrl+]

##### 15.一键到行尾

Fn+右箭头

##### 16.智能提示

Alt+Enter智能提示你选择合适的操作

##### 17.整理代码

Ctrl+Alt+L

##### 18.快速复制当前行或者选中内容

Ctrl+D

##### 19.移动当前行或选中内容进行上下移动

Ctrl+Shift+上下键

##### 20.编码设置

在IDE Encoding、Project Encoding、Property Files三处都使用UTF-8编码，同时在文件头添加

```
#-*- coding: utf-8 -*
```

##### 21.调试

```
调试的基本步骤：确定调试的起始步；点击调试（小虫子）；一步步点击，查看运行状态
简单介绍调试栏的几个重要的按钮作用：
Resume Program：断点调试后，点击按钮，继续执行程序
Step Over:在单步执行时，在函数内遇到子函数时不会进入子函数内单步执行，而是将子函数整个执行完再停止，也就是把子函数整体作为一步。在不存在子函数的情况下是和Step Info效果一样的。
Step Into:单步执行，遇到子函数就进入并且继续单步执行（简言之，进入子函数）
Step Out:当单步执行到子函数内时，用Step out就可以执行完子函数余下部分，并返回到上一层函数。
```

==关于调试的一些链接==

[如何在Python中使用断点调试](https://zhuanlan.zhihu.com/p/21304838)

[pycharm 如何程序运行后，仍可查看变量值？](https://zhuanlan.zhihu.com/p/27062841)

##### 22.替换

Ctrl+R

##### 23.更改配色方案

> 1.打开网址http://www.easycolor.cc/intelliJidea/list.html
>
> 2.下载自己喜欢的方案
>
> 3.在pycharm中：file-import该配色方案的jar包-重启即可

> 1.源文件下载https://github.com/d1ffuz0r/pycharm-themes
>
> 2.复制到C:\Users\23842\.PyCharmCE2018.1\config\colors
>
> 3.在pycharm的file->Setting->Editor->Colors & Fonts里就可以看到刚刚下载的颜色字体主题包