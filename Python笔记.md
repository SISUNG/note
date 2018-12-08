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



