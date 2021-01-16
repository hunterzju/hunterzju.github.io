python中栈的实现[ch3.5](https://runestone.academy/runestone/static/pythonds/BasicDS/ImplementingaStackinPython.html)

## 基本实现

栈是一种抽象的数据结构，特点是先入后出，python中栈的实现可以借助基本的数据结构，需要实现栈的主要方法有：

```python
class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

```

栈采用有序数据结构**列表**实现，需要实现栈空、入栈、出栈、栈深度和返回栈顶元素的方法。

python自带的标准模块中也有栈的实现,可以直接调用：

```python
from pythonds.basic.stack import Stack

s=Stack()

print(s.isEmpty())
s.push(4)
s.push('dog')
print(s.peek())
s.push(True)
print(s.size())
print(s.isEmpty())
s.push(8.4)
print(s.pop())
print(s.pop())
print(s.size())
```

## 应用—— 简单的平衡括号

数据计算中经常遇到这样的表达式：(5+6)∗(7+8)/(4+3)  

还有程序中函数也会遇到很多括号对，只有每一个'('对应一个')'这种表达式才是合法的。

下面利用栈实现了简单地判断括号是否平衡地代码

```python
from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == "(":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()

        index = index + 1

    if balanced and s.isEmpty():
        return True
    else:
        return False

print(parChecker('((()))'))
print(parChecker('(()'))
```

有时候会出现\'[]\'和\'{}\'两种括号，下面代码实现了判断多种括号混合是否平衡地代码：

```python
from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in "([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top,symbol):
                       balanced = False
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)


print(parChecker('{([][])}()}'))
print(parChecker('[{()]'))

```

## 应用——十进制转二进制

计算机中处理数据都是二进制，日常生活中多使用十进制表示，十进制与二进制转换可以看作以下过程：

从十进制233转换为二进制11101001过程

![](./pic/dectobin.png)

采用栈地方式实现：

```python
from pythonds.basic.stack import Stack

def divideBy2(decNumber):
    remstack = Stack()

    while decNumber > 0:
        rem = decNumber % 2
        remstack.push(rem)
        decNumber = decNumber // 2

    binString = ""
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())

    return binString

print(divideBy2(42))
```

从十进制转换为8进制或者十六进制如下：

```python
from pythonds.basic.stack import Stack

def baseConverter(decNumber,base):
    digits = "0123456789ABCDEF"

    remstack = Stack()

    while decNumber > 0:
        rem = decNumber % base
        remstack.push(rem)
        decNumber = decNumber // base

    newString = ""
    while not remstack.isEmpty():
        newString = newString + digits[remstack.pop()]

    return newString

print(baseConverter(25,2))
print(baseConverter(25,16))
```



