##### 1.常用快捷键总结

| 快捷键               | 功能                       | 快捷键                   | 功能                                              |
| -------------------- | -------------------------- | ------------------------ | ------------------------------------------------- |
| Alt+F1               | 查找当前文件               | Ctrl+Shift++             | 快速展开代码                                      |
| Ctrl+Tab             | 切换窗口                   | Ctrl+Shift+-             | 快速收缩代码                                      |
| Ctrl+X               | 剪贴                       | Shift+Delete<br />Ctrl+Y | 快速删除当前行                                    |
| Ctrl+B/Ctrl+鼠标左键 | 看声明                     | Ctrl+Up（Down）          | 窗口上（下）移                                    |
| Ctrl+N               | 查找类名                   | Ctrl+Shift+Up（Down）    | 当前行上（下）移                                  |
| Shift+F9             | 调试                       | 双击Shift                | 超级搜索（搜索目录的技巧是在关键字前面加上斜杠/） |
| Shift+Enter          | 快速换行                   | Ctrl+Shift+F10           | 快速运行代码                                      |
| Ctrl+Shift+V         | 历史粘贴板                 | Ctrl+/                   | 快速注释（或快速取消注释）                        |
| Ctrl+E               | 打开最近访问过的文件       | Alt+V<br />Alt+H         | 纵向分割<br />横向分割                            |
| 选中内容+Tab         | 整行缩进                   | 选中内容+Shift+Tab       | 取消整行缩进                                      |
| Ctrl+[               | 快速定位到代码块开始       | Ctrl+]                   | 快速定位到代码块结尾                              |
| Home                 | 一键到行首                 | End                      | 一键到行尾                                        |
| Ctrl+D               | 快速复制当前行或者选中内容 | Ctrl+R                   | 替换                                              |
| Ctrl+Shift+L         | 快速整理代码               | Ctrl+Insert              | 选中当前行                                        |
| Ctrl+Home            | 快速调到文件头部           | Ctrl+End                 | 快速调到文件尾部                                  |
| Shift+F6             | 更改变量/方法名字          | Ctrl+J                   | 输入模板                                          |
| Ctrl+F12             | 快速调到List               | Ctrl+Q                   | 查看参数；查看文档字符串                          |
| Alt+Left（Right）    | 切换文件                   |                          |                                                   |
|                      |                            |                          |                                                   |



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

24.structure;rename;console;cmd'