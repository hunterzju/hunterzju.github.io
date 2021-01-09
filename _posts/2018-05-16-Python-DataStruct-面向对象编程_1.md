python中面向对象编程，[原文1.13节](https://runestone.academy/runestone/static/pythonds/Introduction/ObjectOrientedProgramminginPythonDefiningClasses.html)

## 1.13.1 一个分数类(Fraction Class)

新定义一个分数类，使得分数可以像其他数据类型一样进行加，减，乘，除等运算操作，并且无论采用什么操作得到的结果应该是最简的形式，此外显示的时候可以按照分数的格式显示出来。

在python中定义一个新的类时，需要提供一个类名以及属于该类的方法，方法的定义和函数类似。新的类一般需要提供一个`__init__()`（初始化）方法，这个方法在生成一个该类的对象时会自动调用。在分数类中，需要初始化分母和分子。示例如下：

```python
class Fraction:

    def __init__(self,top,bottom):

        self.num = top
        self.den = bottom
```

初始化方法中有三个参数(self,top,bottom)。`self`是一个特殊的参数，总是指向对象自身，只有在被调用的时候才会赋值。Fraction类初始化需要分子和分母的值self.num定义该类有一个叫做num的数据对象用于存放分子，self.den定义了用于存放分母的数据对象。

在新建一个Fraction类的实例的时候需要调用上面的构造函数，`__init__`方法在创建类的时候就会被调用，创建实例的示例如下：

```python
myfraction = Fraction(3,5)
```

这条语句创建了一个叫做myfraction()的对象，代表了分数3/5。

![](https://runestone.academy/runestone/static/pythonds/_images/fraction1.png)

接下来考虑抽象数据类型的一些操作，首先是print：

```python
>>> myf = Fraction(3,5)
>>> print(myf)
<__main__.Fraction instance at 0x409b1acc>
```

直接输出该对象，会在屏幕上显示一个类的实例信息，并不是输出3/5这个数值。

有两个方法可以解决这个问题，一个是定义一个show()方法用于输出分数值，但是这需要调用show()方法才能正确显示：

```python
def show(self):
     print(self.num,"/",self.den)

>>> myf = Fraction(3,5)
>>> myf.show()
3 / 5
>>> print(myf)
<__main__.Fraction instance at 0x40bce9ac>
```

另一种方法是重写已经存在的方法，python中所有的对象都有一系列的标准方法，但有时候这些方法不是我们期望的，我们可以通过重写（overrides）已经存在的方法，使之符合期望。其中`__str__`就是一个标准方法，通过重写该方法可以使得Fraction类的示例在需要返回string类型的时候返回我们定义的格式：

```python
def __str__(self):
    return str(self.num)+"/"+str(self.den)
    
>>> myf = Fraction(3,5)
>>> print(myf)
3/5
>>> print("I ate", myf, "of the pizza")
I ate 3/5 of the pizza
>>> myf.__str__()
'3/5'
>>> str(myf)
'3/5'
```

我们还可以重写Fraction类中的很多方法，最重要的就是那些基本的算术操作，比如说加法（add)，在执行"+"操作的时候会调用类的`__add__`方法，可以通过重写该方法使得Fraction类支持分数加法：

```python
def __add__(self,otherfraction):

     newnum = self.num*otherfraction.den + self.den*otherfraction.num
     newden = self.den * otherfraction.den

     return Fraction(newnum,newden)
    
>>> f1=Fraction(1,4)
>>> f2=Fraction(1,2)
>>> f3=f1+f2
>>> print(f3)
6/8
```

这里相加的结果并不是最简的形式，需要用分子分母的最大公约数来化简，求最大公约数(GCD)可以使用欧几里德算法（Euclid），算法实现如下：

```python
def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n

print(gcd(20,10))
```

用最大公约数对求和结果进行化简，重新调整`__add__`函数：

```python
def __add__(self,otherfraction):
    newnum = self.num*otherfraction.den + self.den*otherfraction.num
    newden = self.den * otherfraction.den
    common = gcd(newnum,newden)
    return Fraction(newnum//common,newden//common)

>>> f1=Fraction(1,4)
>>> f2=Fraction(1,2)
>>> f3=f1+f2
>>> print(f3)
3/4
```

![](https://runestone.academy/runestone/static/pythonds/_images/fraction2.png)

现在的Fraction类具有`__add__`和`__str__`两种方法，如果要判断两个分数f1和f2是否相等还是无法实现。可以通过重写`__eq__`方法来实现：

```python
def __eq__(self, other):
    firstnum = self.num * other.den
    secondnum = other.num * self.den

    return firstnum == secondnum
```

下面是Fraction类的总结:

```python
def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n

class Fraction:
     def __init__(self,top,bottom):
         self.num = top
         self.den = bottom

     def __str__(self):
         return str(self.num)+"/"+str(self.den)

     def show(self):
         print(self.num,"/",self.den)

     def __add__(self,otherfraction):
         newnum = self.num*otherfraction.den + \
                      self.den*otherfraction.num
         newden = self.den * otherfraction.den
         common = gcd(newnum,newden)
         return Fraction(newnum//common,newden//common)

     def __eq__(self, other):
         firstnum = self.num * other.den
         secondnum = other.num * self.den

         return firstnum == secondnum

x = Fraction(1,2)
y = Fraction(2,3)
print(x+y)
print(x == y)
```





