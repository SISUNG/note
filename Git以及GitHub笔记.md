### 模块一：《GitHub+Hexo个人博客搭建问题记录》

___

##### 有用链接

[《Hexo+github搭建个人博客》](http://www.cnblogs.com/dantefung/p/d8c48ba8030bcab7cfc364d423186fee.html)

[GitHub+Hexo 搭建个人网站详细教程](https://zhuanlan.zhihu.com/p/26625249)

[常用主题](https://github.com/hexojs/hexo/wiki/Themes)

[next主题](https://github.com/theme-next/hexo-theme-next)

[hexo+next主题优化之加入网易云音乐、网易云跟帖、炫酷动态背景](https://blog.csdn.net/sunshine940326/article/details/69933696)

Q1：GitHub Pages域名太长

A1：设置为sisung.github.io即可

Q2：GitHub与Hexo设置不同步

A2：在_config.yml文件中注意空格的问题，由于无法确认中英文输入法，最好从其它行中复制

Q3：fatal: could not create work tree dir 'themes/apollo': Permission denied

A3：可能是Git Bash的权限问题，幸好CMD可以调用Bash，因此就可以以管理员身份运行CMD，进行主题的下载

Q4：注意站点设置和主题设置这两个主题的区别

A4：:ok:

Q5：如何避免每次提交都需要输入用户名和密码？

A5：SSH

Q6：next主题高级设置

A6：Schemes设置在179-183行	hexo g和hexo d

Q7：发布文章流程

A7：

```
hexo n "zhuxingsheng_cn"
之后就会发现在blog根目录下的source文件夹中的_post文件夹中多了一个zhuxingsheng_cn.md 文件
hexo s --debug 在本地浏览器的localhost:4000 预览博文效果
```

Q8：图床调研

A8：有哪些好用的图床？新浪微博？七牛云？

[知乎-图床](https://www.zhihu.com/search?type=content&q=图床)

[贴图库](http://www.tietuku.com/upload)

[Fork me on Github_ribbons](https://github.blog/2008-12-19-github-ribbons/)

[For me on Github_corners](http://tholman.com/github-corners/)

Q9：页面设置为中文

A9：language: zh-CN

```
这些配置都要与 next 目录下的 languages 文件中对应的 yml 文档里配置相关联。
比如你在 hexo 目录中的配置文件设置 language 为 zh-Hans，那么就要进入到 next 目录下的 languages 文件中修改 zh-Hans.yml。
```

Q10：添加标签页|添加分类

A10：

```

```



tags: [Github,Hexo,blog]，逗号为英文输入法下的逗号
categories: 前端

Q11：图标库

[Font Awesome](http://fontawesome.dashgame.com/)

[easyicon](https://www.easyicon.net/)

如果解析出来之后，你的原始链接有问题，那么在根目录下_config.yml中写成类似这样：

`url: https://sisung.github.io`

Q12：注释

`<!-- and -->`



Q13： 文章加密访问

A13：打开 `themes/next/layout/_partials/head.swig`文件,在 `{% if theme.pace %}` 标签下的 `{% endif %}` 之前插入代码：

Q14：文字添加背景色块

打开 `themes/next/source/css/_custom` 下的 `custom.styl` 文件,添加属性样式：

```
// 颜色块-黄
span#inline-yellow {
display:inline;
padding:.2em .6em .3em;
font-size:80%;
font-weight:bold;
line-height:1;
color:#fff;
text-align:center;
white-space:nowrap;
vertical-align:baseline;
border-radius:0;
background-color: #f0ad4e;
}
// 颜色块-绿
span#inline-green {
display:inline;
padding:.2em .6em .3em;
font-size:80%;
font-weight:bold;
line-height:1;
color:#fff;
text-align:center;
white-space:nowrap;
vertical-align:baseline;
border-radius:0;
background-color: #5cb85c;
}
// 颜色块-蓝
span#inline-blue {
display:inline;
padding:.2em .6em .3em;
font-size:80%;
font-weight:bold;
line-height:1;
color:#fff;
text-align:center;
white-space:nowrap;
vertical-align:baseline;
border-radius:0;
background-color: #2780e3;
}
// 颜色块-紫
span#inline-purple {
display:inline;
padding:.2em .6em .3em;
font-size:80%;
font-weight:bold;
line-height:1;
color:#fff;
text-align:center;
white-space:nowrap;
vertical-align:baseline;
border-radius:0;
background-color: #9954bb;
}
```

在你需要编辑的文章地方。放置如下代码：

```text
<span id="inline-blue"> 站点配置文件 </span>
<span id="inline-purple"> 主题配置文件 </span>
<span id="inline-yellow"> 站点配置文件 </span>
<span id="inline-green"> 主题配置文件 </span>
```

Q15：页脚访问人数不显示

[hexo页脚添加访客人数和总访问量](https://www.jianshu.com/p/c311d31265e0)

Q16：文档加密阅读

[某大神](https://lewky.cn/posts/15308.html)

把head.swig复制出来试试看（还是不行）

[一只老菜鸡](https://www.lee1224.com/bdseo/)

Q17：

INFO  Start processing
FATAL Something's wrong. Maybe you can find the solution here: http://hexo.io/docs/troubleshooting.html
TypeError: Cannot read property 'count' of undefined

A17：

1.npm remove hexo-baidu-url-submit
2.hexo clean
3.hexo g
You can try it.
hope that can help you.

