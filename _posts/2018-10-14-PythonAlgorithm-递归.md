# python算法——递归

递归算法的三条重要准则：

* 递归算法必须含有一个基本情况**base case**（作为递归终止条件）
* 递归算法中每一次递归必须改变当前状态并向基本情况靠近
* 递归算法必须递归调用本身

## 示例1——将整数转化为任意进制的字符串表示

比如说整数10可以表示为十进制的`"10"`，也可以表示为二进制的`"1010"`。

我们首先以十进制的情况为例：将整数769转换为十进制的字符串表示形式。

可以观察到，如果有一个字符串表`convString = "0123456789"`，只需要将769中的各位拿出来，通过查表即可实现：`convString[7]`对应字符`"7"`，其余各位也是类似的。

那么，上述问题可以通过以下三步解决：

* 从原来的整数中取出单独一位
* 将取出的一位通过查表转换出来
* 将所有单独的位连接起来

接下来的问题就是如何取出单独的一位：通过整除的方式即可单独取出各位。比如：

769 // 10 = 76

769 % 10 = 9

整除之后，商为76，余数为9，就可以将末尾单独一位取出来。

这样就可以实现任意进制的转换：

```python
def toStr(n,base):
   convertString = "0123456789ABCDEF"
   if n < base:
      return convertString[n]
   else:
      return toStr(n//base,base) + convertString[n%base]

print(toStr(1453,16))
```

## 示例2——用栈实现进制转换

示例1采用了递归的方式实现进制转换，如果要使用非递归的方法，可以用栈来实现：

```python
from pythonds.basic.stack import Stack

rStack = Stack()

def toStr(n,base):
    convertString = "0123456789ABCDEF"
    while n > 0:
        if n < base:
            rStack.push(convertString[n])
        else:
            rStack.push(convertString[n % base])
        n = n // base
    res = ""
    while not rStack.isEmpty():
        res = res + str(rStack.pop())
    return res

print(toStr(1453,16))
```

该示例可以帮助我们了解python中递归的实现，其实再示例1中每一次调用`toStr()`函数的时候，会将当前函数的上下文(context)压入栈中，上下文中包括了栈指针和函数局部变量等内容；当最后一次`toStr()`执行返回结果后，之前栈里的函数会依次出栈进行计算，此时所有的函数都可以得到结果了。

## 实例3——采用turtle递归绘制金字塔

采用递归的方法绘制一座层层包含的三角金字塔：

```python
import turtle

def drawTriangle(points,color,myTurtle):
    myTurtle.fillcolor(color)
    myTurtle.up()
    myTurtle.goto(points[0][0],points[0][1])
    myTurtle.down()
    myTurtle.begin_fill()
    myTurtle.goto(points[1][0],points[1][1])
    myTurtle.goto(points[2][0],points[2][1])
    myTurtle.goto(points[0][0],points[0][1])
    myTurtle.end_fill()

def getMid(p1,p2):
    return ( (p1[0]+p2[0]) / 2, (p1[1] + p2[1]) / 2)

def sierpinski(points,degree,myTurtle):
    colormap = ['blue','red','green','white','yellow',
                'violet','orange']
    drawTriangle(points,colormap[degree],myTurtle)
    if degree > 0:
        sierpinski([points[0],
                        getMid(points[0], points[1]),
                        getMid(points[0], points[2])],
                   degree-1, myTurtle)
        sierpinski([points[1],
                        getMid(points[0], points[1]),
                        getMid(points[1], points[2])],
                   degree-1, myTurtle)
        sierpinski([points[2],
                        getMid(points[2], points[1]),
                        getMid(points[0], points[2])],
                   degree-1, myTurtle)

def main():
   myTurtle = turtle.Turtle()
   myWin = turtle.Screen()
   myPoints = [[-100,-50],[0,100],[100,-50]]
   sierpinski(myPoints,4,myTurtle)
   myWin.exitonclick()

main()

```

因为颜色限制，代码中金字塔最多可以有7层，`sierpinski(points,degree,myTurtle)`函数中`degree`参数最大可以到6（从0-6，供七层）。