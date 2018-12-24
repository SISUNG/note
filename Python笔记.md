##### 1.os.path

```python
os.path.abspath(path)#返回绝对路径
os.path.basename(path)#返回文件名
os.path.commonprefix(list)#返回list（多个路径）中所有path共有的最长路径
os.path.dirname(path)#返回文件路径
os.path.exists(path)  #路径存在则返回True,路径损坏返回False
os.path.lexists  #路径存在则返回True,路径损坏也返回True
os.path.expanduser(path)  #把path中包含的"~"和"~user"转换成用户目录
os.path.expandvars(path)  #根据环境变量的值替换path中包含的”$name”和”${name}”
os.path.getatime(path)  #返回最后一次进入此path的时间。
os.path.getmtime(path)  #返回在此path下最后一次修改的时间。
os.path.getctime(path)  #返回path的大小
os.path.getsize(path)  #返回文件大小，如果文件不存在就返回错误
os.path.isabs(path)  #判断是否为绝对路径
os.path.isfile(path)  #判断路径是否为文件
os.path.isdir(path)  #判断路径是否为目录
os.path.islink(path)  #判断路径是否为链接
os.path.ismount(path)  #判断路径是否为挂载点（）
os.path.join(path1[, path2[, ...]])  #把目录和文件名合成一个路径
os.path.normcase(path)  #转换path的大小写和斜杠
os.path.normpath(path)  #规范path字符串形式
os.path.realpath(path)  #返回path的真实路径
os.path.relpath(path[, start])  #从start开始计算相对路径
os.path.samefile(path1, path2)  #判断目录或文件是否相同
os.path.sameopenfile(fp1, fp2)  #判断fp1和fp2是否指向同一文件
os.path.samestat(stat1, stat2)  #判断stat tuple stat1和stat2是否指向同一个文件
os.path.split(path)  #把路径分割成dirname和basename，返回一个元组
os.path.splitdrive(path)   #一般用在windows下，返回驱动器名和路径组成的元组
os.path.splitext(path)  #分割路径，返回路径名和文件扩展名的元组
os.path.splitunc(path)  #把路径分割为加载点与文件
os.path.walk(path, visit, arg)  #遍历path，进入每个目录都调用visit函数，visit函数必须有3个参数(arg, dirname, names)，dirname表示当前目录的目录名，names代表当前目录下的所有文件名，args则为walk的第三个参数os.path.supports_unicode_filenames  #设置是否支持unicode路径名
```



##### 2.__ init __方法

> 可以直接理解成在类初始化的时候，创造或者带入一个或者几个类的全局变量。



##### 3.类对象、实例对象、类变量、实例变量、类方法、实例方法、静态方法各个概念的解析

```python
类对象和实例对象
#Python中一切皆为对象，类本身也是一种对象，类定义完成后，会在当前作用域中定义一个以类名为名字的命名空间。类对象具有以下两种操作：一是可以通过“类名（）”的方式实例化一个对象；二是可以通过“类名.类属性”的方式来访问一个类属性。如果说类是一种概念性的定义，是一种类别，那么实例对象就是对这一类别的具体化、实例化，即实例化对象是类对象实例化之后的产物。

类变量和实例变量
#概念上的区别：类变量是指该类的所有实例所共有的数据，实例变量是该类每一个实例所特有的数据
class Person:
    move = True #这是类变量
	def __ init__(self, name, age):
        self.name = name #这是实例变量
        self.age = age #这是实例变量
        
        
#声明上的区别：类变量声明通常是在类内部，但是在函数体外，不需要用任何关键字修饰。实例变量一般声明在实例方法内部，且用self关键字修饰
class Person:
    move = True    # 这是类变量，

    def __init__(self , name):
        self.name = name  # 这是实例变量，必须声明在实例函数内，用self关键字修饰
        # move = True  # 类变量不能再函数体内声明，在这个位置声明的又没有self关键字修饰，只能是一个局部变量
        
    # self.age = age  # 这是错误的，实例变量不能在这里声明

　　 eat = True  # 这是类变量，可以在函数体外，类内部任意位置
		#上面的变量绑定都是在对象声明过程中绑定的，但事实上类变量和实例变量都可以在类或者实例都可以在对象声明结束之后再绑定，“类名.变量名”绑定的是类变量，"实例名.变量名"绑定的是实例变量。虽然可以在对象声明之后再绑定对象，但是这种方式最好不要使用。
class Person:
    move = True   

    def __init__(self , name , age):
        self.name = name 
        self.age = age 

p1 = Person(‘张三‘ , 20)
p1.gender=‘男‘ # 声明实例对象结束之后再绑定一个实例变量
Person.eat = True  # 声明类对象结束之后再绑定一个类变量
print(p1.gender)  # 输出结果：男
print(p1.eat)  #输出结果：True


#访问上的区别：类变量可以通过“类名.变量名”和“实例名.变量名”的方式访问。实例变量只能通过“实例名.变量名”的方式来访问。虽然可以通过“实例名.类变量名”的方式来访问类变量，但是并不推荐，最好还是通过“类名.类变量名”来访问类变量。
class Person:
    move = True    # 这是类变量
    
    def __init__(self , name , age):
        self.name = name  # 这是实例变量
        self.age = age  # 这是实例变量
        
p1 = Person(‘张三‘ , 20)
print(p1.name  , p1.age) # 通过“实例名.变量名”的方式访问实例变量
print(p1.move) # 通过“实例名.变量名”的方式访问实例变量
print(Person.move)  # 通过“类名.变量名”方式访问类变量
	# print(Person.name) # 这是错误的
 

#存储上的区别：类变量只会在用class关键字声明一个类时创建，且会保存在类的命名空间中，这个类的实例的命名空间中是没有的。通过“实例名.类变量名”访问类变量时，实际访问的是类命名空间中的数据，所以所有实例访问到的数据都是同一个变量，实例变量保存在实例各自的命名空间中。可以通过id（）函数访问内存地址.

静态方法
静态方法是指在定义时，使用@staticmethod装饰器来修饰，无需传入self或cls关键字即可创建的方法。在调用过程中。无需将类实例化，直接通过“类名.方法名（）”方式调用方法。当然，也可以在实例化后通过“实例名.方法名（）”的方式来调用。在静态方法内部，只能通过“类名.类变量名”的方式访问类变量。
class Person:
    move = True
    def __init__(self , name , age):
        self.name = name
        self.age = age
    @staticmethod
    def static_fun(): # 声明一个静态方法
        print(Person.move)
p1 = Person(‘张三‘ , 20)
p1.static_fun() #输出结果：这是静态方法
Person.static_fun() #输出结果：这是静态方法


类方法
类方法需要使用@classmethod装饰器来修饰，且传入的第一个参数为cls，指代的是类本身。类方法在调用方式上与静态方法相似，即可以通过“类名.方法名（）”和“实例名.方法名（）”两种方式调用。但类方法与静态方法不同的是，类方法可以在方法内部通过cls关键字访问类变量。在类方法内部，既能通过“类名.类变量名”的方式访问类变量，也能通过“cls.类变量名”的方式访问类变量。
class Person:
    move = True
    def __init__(self , name , age):
        self.name = name
        self.age = age

    @classmethod
    def class_fun(cls): # 声明一个类方法
        print(cls.move)
        print(Person.move)# cls 指的就是Person类，等效

p1 = Person(‘张三‘ , 20)
p1.class_fun() #输出结果：True True
Person.class_fun() #输出结果：True True


实例方法
在一个类中，除了静态方法和类方法之外，就是实例方法了，实例方法不需要装饰器修饰，不过在声明时传入的第一个参数必须为self，self指代的就是实例本身。实例方法能访问实例变量，静态方法和类方法则不能。在实例方法内部只能通过“类名.类变量名”的方式访问类变量。在调用时，实例方法可以通过“实例名.实例方法名”来调用，如果要通过类来调用，必须显式地将实例当做参数传入。
class Person:
    move = True

    def __init__(self , name , age):
        self.name = name
        self.age = age

    def instance_fun(self): # 声明一个实例方法
        print(Person.move) # 访问类变量
        print(self.name , self.age)

p1 = Person(‘张三‘ , 20)
p1.instance_fun()
Person.instance_fun(p1) #通过类访问实例方法时，必须显式地将实例当做参数传入
```



##### 4.super()方法的使用

如果在子类中也定义了构造器，即__ init __()函数，那么基类的构造器该如何调用呢？

```python
方法一：明确指定，在子类的构造器中明确地指明调用基类的构造器
class C(P):
    def __init__(self):
        P.__init__(self)

方法二：使用super()方法，该方法的漂亮之处在于不需要在定义子类构造器的时候，明确指定子类的基类并显式地调用，即不需要明确地提供父类，这样做的好处就是如果你改变了继承的父类，你只需要修改一行代码，而不需要在大量代码中查找那个要修改的基类，另外一方面代码的可移植性和重用性也更高。
class C(P):
    def __init__(self):
        super(C，self).__init__()
```



##### 5.os模块

官方解释：This module provides a portable way of using operating system dependent functionality，该模块提供了一种方便的使用操作系统函数的方法。也就是说，os模块负责程序与操作系统的交互，提供了访问操作系统底层的接口。

```python
os常用方法
os.remove()#删除文件
os.rename()#重命名文件
os.walk()#生成目录树下的所有文件名
os.chdir()#改变目录
os.mkdir()/os.mkdirs()#创建目录、创建多层目录
os.rmdir()/os.rmdirs()#删除目录、删除多层目录
os.listdir()#列出指定目录的文件
os.getcwd()#返回当前工作目录
os.chmod()#改变目录权限
os.path.basename() #去掉目录路径，返回文件名
os.path.dirname() #去掉文件名，返回目录路径
os.path.join() #将分离的各部分组合成一个路径名
os.path.split() #返回( dirname(), basename())元组
os.path.splitext() #返回 (filename, extension) 元组
os.path.getatime\ctime\mtime #分别返回最近访问、创建、修改时间
os.path.getsize() #返回文件大小
os.path.exists() #是否存在
os.path.isabs() #是否为绝对路径
os.path.isdir() #是否为目录
os.path.isfile() #是否为文件
os.system('cmd') #可以直接在python里调用第三方程序，如matlab，notepad++，在写脚本的时候很方便。
os.access(path,mode) #检验权限模式
os.chflags(path,flags) #设置路径的标记为数字标记
```



##### 6.sys模块

官方解释：This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.这个模块可供访问由解释器访问或维护的变量和与解释器进行交互的函数。也就是说，sys模块负责程序与Python解释器的交互，提供了一系列的函数和变量，用于操纵python的运行时环境。

> import sys
>
> dir(sys) #可以通过dir()方法查看模块中可用的方法。

```python
sys.argv #命令行参数List，第一个元素是程序本身路径
sys.modules.keys() #返回所有已经导入的模块列表
sys.exc_info() #获取当前正在处理的异常类,exc_type、exc_value、exc_traceback当前处理的异常详细信息
sys.exit(n) #退出程序，正常退出时exit(0)。当然也可以用字符串参数，表示错误不成功的报错信息。
sys.hexversion #获取Python解释程序的版本值，16进制格式如：0x020403F0
sys.version #获取Python解释程序的版本信息
sys.maxint #最大的Int值
sys.maxunicode #最大的Unicode值
sys.modules #返回系统导入的模块字段，key是模块名，value是模块
sys.path #返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
sys.path.append('自定义模块路径') #自定义添加模块路径
sys.platform #返回操作系统平台名称

sys.stdout #标准输出
#print('Hello World!\n')等效于sys.stdout.write('Hello World!\n')

sys.stdin #标准输入
#input('Please enter your name：')等效于sys.stdin.readline()[:-1]

sys.stderr #错误输出
sys.exc_clear() #用来清除当前线程所出现的当前的或最近的错误信息
sys.exec_prefix #返回平台独立的python文件安装的位置
sys.byteorder #本地字节规则的指示器，big-endian平台的值是'big',little-endian平台的值是'little'
sys.copyright #记录python版权相关的东西
sys.api_version #解释器的C的API版本
sys.modules #其是一个全局字典，该字典是python启动后就加载在内存中，每当程序员导入新的模块，sys.modules将自动记录该模块。当第二次再导入该模块时，python会直接在字典中查找，从而加快了程序运行的速度，它拥有字典所拥有的一切方法。
sys.getdefaultencoding()#获取系统当前编码，一般默认为ASCII
sys.setdefaultencoding()#设置系统默认编码，执行dir（sys）时不会看到这个方法，在解释器中执行不通过，可以先执行reload(sys)，再执行 setdefaultencoding(‘utf8’)，此时将系统默认编码设置为'utf-8'。
sys.getfilesystemencoding() #获取文件系统使用编码方式，Windows下返回’mbcs’，mac下返回’utf-8’

import sys
def exitfunc(value):
    print (value)
    sys.exit(0)
print("hello")
try:
    sys.exit(90)
except SystemExit as value:
    exitfunc(value)   
print("come?")
运行结果
hello
90
#程序首先打印hello，再执行exit（90），抛异常把90传给values，values在传进函数中执行，打印90，程序退出。后面的”come?”因为已经退出所以不会被打印. 而此时如果把exitfunc函数里面的sys.exit(0)去掉,那么程序会继续执行到输出”come?”。
```



##### 7.处理日期和时间的模块：time模块、calendar模块、datetime模块、pytz模块、dateutil模块

- [x] time模块

```python
获取当前时间戳 time.time() #时间戳单位最适合用于做日期运算。但是1970年之前的日期就无法用此表示了，太遥远的日期也不行，Unix和Windows只支持到2038年。

时间元组 #字段(4位数年，月，日，时，分，秒，一周的第几日，一年的第几日，夏令时)
		#属性（tm_year,tm_mon,tm_mday,tm_hour,tm_min,tm_sec,tm_wday,tm_yday,tm_isdst)
    
获取当前时间 time.localtime(time.time())返回一个时间元组

获取格式化的时间 time.asctime(time.localtime(time.time()))

格式化日期 time.strftime(format[,t])
# 格式化成2016-03-20 11:45:39形式
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
# 格式化成Sat Mar 28 22:24:24 2016形式
print time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  
# 将格式字符串转换为时间戳
a = "Sat Mar 28 22:24:24 2016"
print time.mktime(time.strptime(a,"%a %b %d %H:%M:%S %Y"))

python中时间日期格式化符号：
%y 两位数的年份表示（00-99）
%Y 四位数的年份表示（000-9999）
%m 月份（01-12）
%d 月内中的一天（0-31）
%H 24小时制小时数（0-23）
%I 12小时制小时数（01-12）
%M 分钟数（00=59）
%S 秒（00-59）
%a 本地简化星期名称
%A 本地完整星期名称
%b 本地简化的月份名称
%B 本地完整的月份名称
%c 本地相应的日期表示和时间表示
%j 年内的一天（001-366）
%p 本地A.M.或P.M.的等价符
%U 一年中的星期数（00-53）星期天为星期的开始
%w 星期（0-6），星期天为星期的开始
%W 一年中的星期数（00-53）星期一为星期的开始
%x 本地相应的日期表示
%X 本地相应的时间表示
%Z 当前时区的名称
%% %号本身
```

- [ ] calendar模块
- [x] datetime模块

```python
datetime.date.today() 本地日期对象,(用str函数可得到它的字面表示(2014-03-24))
datetime.date.isoformat(obj) 当前[年-月-日]字符串表示(2014-03-24)
datetime.date.fromtimestamp() 返回一个日期对象，参数是时间戳,返回 [年-月-日]
datetime.date.weekday(obj) 返回一个日期对象的星期数,周一是0
datetime.date.isoweekday(obj) 返回一个日期对象的星期数,周一是1
datetime.date.isocalendar(obj) 把日期对象返回一个带有年月日的元组
datetime对象：
datetime.datetime.today() 返回一个包含本地时间(含微秒数)的datetime对象 2014-03-24 23:31:50.419000
datetime.datetime.now([tz]) 返回指定时区的datetime对象 2014-03-24 23:31:50.419000
datetime.datetime.utcnow() 返回一个零时区的datetime对象
datetime.fromtimestamp(timestamp[,tz]) 按时间戳返回一个datetime对象，可指定时区,可用于strftime转换为日期表示 
datetime.utcfromtimestamp(timestamp) 按时间戳返回一个UTC-datetime对象
datetime.datetime.strptime(‘2014-03-16 12:21:21‘,”%Y-%m-%d %H:%M:%S”) 将字符串转为datetime对象
datetime.datetime.strftime(datetime.datetime.now(), ‘%Y%m%d %H%M%S‘) 将datetime对象转换为str表示形式
datetime.date.today().timetuple() 转换为时间戳datetime元组对象，可用于转换时间戳
datetime.datetime.now().timetuple()
time.mktime(timetupleobj) 将datetime元组对象转为时间戳
```



- [ ] pytz模块
- [ ] dateutil模块



##### 8.math模块与cmath模块

```python
ceil:取大于等于x的最小的整数值，如果x是一个整数，则返回x
copysign:把y的正负号加到x前面，可以使用0
cos:求x的余弦，x必须是弧度
degrees:把x从弧度转换成角度
e:表示一个常量
exp:返回math.e,也就是2.71828的x次方
expm1:返回math.e的x(其值为2.71828)次方的值减１
fabs:返回x的绝对值
factorial:取x的阶乘的值
floor:取小于等于x的最大的整数值，如果x是一个整数，则返回自身
fmod:得到x/y的余数，其值是一个浮点数
frexp:返回一个元组(m,e),其计算方式为：x分别除0.5和1,得到一个值的范围
fsum:对迭代器里的每个元素进行求和操作
gcd:返回x和y的最大公约数
hypot:如果x是不是无穷大的数字,则返回True,否则返回False
isfinite:如果x是正无穷大或负无穷大，则返回True,否则返回False
isinf:如果x是正无穷大或负无穷大，则返回True,否则返回False
isnan:如果x不是数字True,否则返回False
ldexp:返回x*(2**i)的值
log:返回x的自然对数，默认以e为基数，base参数给定时，将x的对数返回给定的base,计算式为：log(x)/log(base)
log10:返回x的以10为底的对数
log1p:返回x+1的自然对数(基数为e)的值
log2:返回x的基2对数
modf:返回由x的小数部分和整数部分组成的元组
pi:数字常量，圆周率
pow:返回x的y次方，即x**y
radians:把角度x转换成弧度
sin:求x(x为弧度)的正弦值
sqrt:求x的平方根
tan:返回x(x为弧度)的正切值
trunc:返回x的整数部分
```



##### 9.re模块

```python
一.常用正则表达式符号和语法：
'.' 匹配所有字符串，除\n以外
‘-’ 表示范围[0-9]
'*' 匹配前面的子表达式零次或多次。要匹配 * 字符，请使用 \*。
'+' 匹配前面的子表达式一次或多次。要匹配 + 字符，请使用 \+
'^' 匹配字符串开头
‘$’ 匹配字符串结尾 re
'\' 转义字符， 使后一个字符改变原来的意思，如果字符串中有字符*需要匹配，可以\*或者字符集[*] re.findall(r'3\*','3*ds')结['3*']
'*' 匹配前面的字符0次或多次 re.findall("ab*","cabc3abcbbac")结果：['ab', 'ab', 'a']
‘?’ 匹配前一个字符串0次或1次 re.findall('ab?','abcabcabcadf')结果['ab', 'ab', 'ab', 'a']
'{m}' 匹配前一个字符m次 re.findall('cb{1}','bchbchcbfbcbb')结果['cb', 'cb']
'{n,m}' 匹配前一个字符n到m次 re.findall('cb{2,3}','bchbchcbfbcbb')结果['cbb']
'\d' 匹配数字，等于[0-9] re.findall('\d','电话:10086')结果['1', '0', '0', '8', '6']
'\D' 匹配非数字，等于[^0-9] re.findall('\D','电话:10086')结果['电', '话', ':']
'\w' 匹配字母和数字，等于[A-Za-z0-9] re.findall('\w','alex123,./;;;')结果['a', 'l', 'e', 'x', '1', '2', '3']
'\W' 匹配非英文字母和数字,等于[^A-Za-z0-9] re.findall('\W','alex123,./;;;')结果[',', '.', '/', ';', ';', ';']
'\s' 匹配空白字符 re.findall('\s','3*ds \t\n')结果[' ', '\t', '\n']
'\S' 匹配非空白字符 re.findall('\s','3*ds \t\n')结果['3', '*', 'd', 's']
'\A' 匹配字符串开头
'\Z' 匹配字符串结尾
'\b' 匹配单词的词首和词尾，单词被定义为一个字母数字序列，因此词尾是用空白符或非字母数字符来表示的
'\B' 与\b相反，只在当前位置不在单词边界时匹配
'(?P<name>...)' 分组，除了原有编号外在指定一个额外的别名 re.search("(?P<province>[0-9]{4})(?P<city>[0-9]{2})(?P<birthday>[0-9]{8})","371481199306143242").groupdict("city") 结果{'province': '3714', 'city': '81', 'birthday': '19930614'}
[] 是定义匹配的字符范围。比如 [a-zA-Z0-9] 表示相应位置的字符要匹配英文字符和数字。[\s*]表示空格或者*号。

二.常用的re函数：
方法/属性 作用
re.match(pattern, string, flags=0) 从字符串的起始位置匹配，如果起始位置匹配不成功的话，match()就返回none
re.search(pattern, string, flags=0) 扫描整个字符串并返回第一个成功的匹配
re.findall(pattern, string, flags=0) 找到RE匹配的所有字符串，并把他们作为一个列表返回
re.finditer(pattern, string, flags=0) 找到RE匹配的所有字符串，并把他们作为一个迭代器返回
re.sub(pattern, repl, string, count=0, flags=0) 替换匹配到的字符串
```



##### 10.urllib模块

```python
urllib.quote(string[,safe]) 对字符串进行编码。参数safe指定了不需要编码的字符
urllib.unquote(string) 对字符串进行解码
urllib.quote_plus(string[,safe]) 与urllib.quote类似，但这个方法用‘+‘来替换‘ ‘，而quote用‘%20‘来代替‘ ‘
urllib.unquote_plus(string ) 对字符串进行解码
urllib.urlencode(query[,doseq]) 将dict或者包含两个元素的元组列表转换成url参数。
例如 字典{‘name‘:‘wklken‘,‘pwd‘:‘123‘}将被转换为”name=wklken&pwd=123″
urllib.pathname2url(path) 将本地路径转换成url路径
urllib.url2pathname(path) 将url路径转换成本地路径
urllib.urlretrieve(url[,filename[,reporthook[,data]]]) 下载远程数据到本地
filename：指定保存到本地的路径（若未指定该，urllib生成一个临时文件保存数据）
reporthook：回调函数，当连接上服务器、以及相应的数据块传输完毕的时候会触发该回调
data：指post到服务器的数据
rulrs = urllib.urlopen(url[,data[,proxies]]) 抓取网页信息，[data]post数据到Url,proxies设置的代理
urlrs.readline() 跟文件对象使用一样
urlrs.readlines() 跟文件对象使用一样
urlrs.fileno() 跟文件对象使用一样
urlrs.close() 跟文件对象使用一样
urlrs.info() 返回一个httplib.HTTPMessage对象，表示远程服务器返回的头信息
urlrs.getcode() 获取请求返回状态HTTP状态码
urlrs.geturl() 返回请求的URL
```



##### 11.string模块

```python
str.capitalize() 把字符串的第一个字符大写
str.center(width) 返回一个原字符串居中，并使用空格填充到width长度的新字符串
str.ljust(width) 返回一个原字符串左对齐，用空格填充到指定长度的新字符串
str.rjust(width) 返回一个原字符串右对齐，用空格填充到指定长度的新字符串
str.zfill(width) 返回字符串右对齐，前面用0填充到指定长度的新字符串
str.count(str,[beg,len]) 返回子字符串在原字符串出现次数，beg,len是范围
str.decode(encodeing[,replace]) 解码string,出错引发ValueError异常
str.encode(encodeing[,replace]) 解码string
str.endswith(substr[,beg,end]) 字符串是否以substr结束，beg,end是范围
str.startswith(substr[,beg,end]) 字符串是否以substr开头，beg,end是范围
str.expandtabs(tabsize = 8) 把字符串的tab转为空格，默认为8个
str.find(str,[stat,end]) 查找子字符串在字符串第一次出现的位置，否则返回-1
str.index(str,[beg,end]) 查找子字符串在指定字符中的位置，不存在报异常
str.isalnum() 检查字符串是否以字母和数字组成，是返回true否则False
str.isalpha() 检查字符串是否以纯字母组成，是返回true,否则false
str.isdecimal() 检查字符串是否以纯十进制数字组成，返回布尔值
str.isdigit() 检查字符串是否以纯数字组成，返回布尔值
str.islower() 检查字符串是否全是小写，返回布尔值
str.isupper() 检查字符串是否全是大写，返回布尔值
str.isnumeric() 检查字符串是否只包含数字字符，返回布尔值
str.isspace() 如果str中只包含空格，则返回true,否则FALSE
str.title() 返回标题化的字符串（所有单词首字母大写，其余小写）
str.istitle() 如果字符串是标题化的(参见title())则返回true,否则false
str.join(seq) 以str作为连接符，将一个序列中的元素连接成字符串
str.split(str=‘‘,num) 以str作为分隔符，将一个字符串分隔成一个序列，num是被分隔的字符串
str.splitlines(num) 以行分隔，返回各行内容作为元素的列表
str.lower() 将大写转为小写
str.upper() 转换字符串的小写为大写
str.swapcase() 翻换字符串的大小写
str.lstrip() 去掉字符左边的空格和回车换行符
str.rstrip() 去掉字符右边的空格和回车换行符
str.strip() 去掉字符两边的空格和回车换行符
str.partition(substr) 从substr出现的第一个位置起，将str分割成一个3元组。
str.replace(str1,str2,num) 查找str1替换成str2，num是替换次数
str.rfind(str[,beg,end]) 从右边开始查询子字符串
str.rindex(str,[beg,end]) 从右边开始查找子字符串位置 
str.rpartition(str) 类似partition函数，不过从右边开始查找
str.translate(str,del=‘‘) 按str给出的表转换string的字符，del是要过虑的字符
```



##### 12.random模块

```python
random.random() 产生0-1的随机浮点数
random.uniform(a, b) 产生指定范围内的随机浮点数
random.randint(a, b) 产生指定范围内的随机整数
random.randrange([start], stop[, step]) 从一个指定步长的集合中产生随机数
random.choice(sequence) 从序列中产生一个随机数
random.shuffle(x[, random]) 将一个列表中的元素打乱
random.sample(sequence, k) 从序列中随机获取指定长度的片断
```



##### 13.stat模块

```python
描述os.stat()返回的文件属性列表中各值的意义
fileStats = os.stat(path) 获取到的文件属性列表
fileStats[stat.ST_MODE] 获取文件的模式
fileStats[stat.ST_SIZE] 文件大小
fileStats[stat.ST_MTIME] 文件最后修改时间
fileStats[stat.ST_ATIME] 文件最后访问时间
fileStats[stat.ST_CTIME] 文件创建时间
stat.S_ISDIR(fileStats[stat.ST_MODE]) 是否目录
stat.S_ISREG(fileStats[stat.ST_MODE]) 是否一般文件
stat.S_ISLNK(fileStats[stat.ST_MODE]) 是否连接文件
stat.S_ISSOCK(fileStats[stat.ST_MODE]) 是否COCK文件
stat.S_ISFIFO(fileStats[stat.ST_MODE]) 是否命名管道
stat.S_ISBLK(fileStats[stat.ST_MODE]) 是否块设备
stat.S_ISCHR(fileStats[stat.ST_MODE]) 是否字符设置
```



##### 14.内置函数总结

```python
abs() #返回数字的绝对值
bin() #返回一个整数int或者长整数long int的二进制表示的字符串
bool() #用于将给定参数转换为布尔类型，如果没有参数，返回false
chr() #用一个范围在range(256)内的整数作为参数，返回一个对应的字符。参数可以是10进制也可以是16进制的形式的数字，返回值是当前整数对应的ASCII字符。
cmp(x,y) #用于比较2个对象，如果x<y返回-1，如果x==y返回0，如果x>y返回1。
delattr() #用于删除属性。delattr(x,'foobar')相当于del x.foobar
dir() #该函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；带参数时，返回参数的属性、方法列表。如果参数包含方法__dir__(),该方法将被调用。如果参数不包含__dir__(),该方法将最大限度地收集参数信息。
divmod() #该函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组（a//b,a % b)
enumerate() #该函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中。语法为enumerate（sequence，[start=0]),sequence表示一个序列、迭代器或其他支持迭代对象，start表示下标起始位置。
eval() #用来执行一个字符串表达式，并返回表达式的值。eval(expression[, globals[, locals]])。expression为表达式，globals为变量作用域，全局命名空间，如果被提供，则必须是一个字典对象。locals为变量作用域，局部命名空间，如果被提供，可以是任何映射对象。将字符串str当成有效的表达式来求值并返回计算结果，可以把list,tuple,dict和string相互转化
filter() #对sequence中的item依次执行function(item),将执行结果为True的item组成一个List、String、Tuple进行返回
id() #用于获取对象的内存地址
int() #用于将一个字符串或数字转换为整型。class int(x, base=10)，x表示字符串或数字，base表示进制数，默认为十进制。
lambda() #这是python支持的一种有趣的语法，它允许你快速定义单行的最小函数，类似于C语言中的宏，这些叫做lambda的函数，是从LISP借用来的，可以用在任何需要函数的地方。
list() #将元组转换为列表。
map() #map(f,iterable)基本上等于，[f(x) for x in iterable]
ord() #ord() 函数是 chr() 函数（对于8位的ASCII字符串）或 unichr() 函数（对于Unicode对象）的配对函数，它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值，如果所给的 Unicode 字符超出了你的 Python 定义范围，则会引发一个 TypeError 的异常。
oct() #将一个整数转换成8进制字符串。
reverse() #用于反向列表中的元素。
reduce(function,sequence,starting_value) #对sequence中的item顺讯迭代调用function，如果有starting_value，还可以作为初始值调用，例如可以用来对List求和。
vars() #返回对象object的属性和属性值的字典对象。如果没有参数，就打印当前调用位置的属性和属性值，类似locals()。
string() #将对象转化为适于人阅读的形式。
set() #创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
sum() #该方法对列表进行求和计算。sum(iterable[, start])，iterable为可迭代对象，如：列表、元组、集合。start为指定相加的参数，如果没有设置这个值，默认为0。
sorted() #对所有可迭代的对象进行排序操作。sorted(iterable[, cmp[, key[, reverse]]]),iterable为可迭代对象，cmp为比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。key为主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。reverse为排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
tuple() #将列表转换为元组
```



##### 15.logging模块

```python
logging.info()
logging.basicConfig()
logging.error()
```



##### 16.importlib模块

```
importlib.import_module()
```



##### 17.collections模块

```python
from collections import OrderedDict
```



##### 18.pickle模块

##### 19.Python字符串前面加u、r、b的含义

##### 20.glob模块

##### 21.shutil模块

##### 22.PIL模块

##### 23.闭包

> 一个函数和它的环境变量合在一起，就构成了一个闭包(closure)
>
> 在python中，所谓的闭包是一个包含有环境变量取值的函数对象
>
> python中的闭包从表现形式上定义为：如果在一个内部函数里，对在外部作用域（但不是在全局作用域）的变量进行引用，那么内部函数就被认为是闭包（closure）
>
> 闭包=函数块+定义函数时的变量
>
> 闭包中是不能修改外部作用域的局部变量的

##### 24.NameError: name 'reduce' is not defined

> reduce函数在python3的内建函数移除了，放入了functools模块
>
> from functools import reduce