[python数据结构和算法1.10](https://runestone.academy/runestone/static/pythonds/Introduction/ControlStructures.html)

在算法实现中，**迭代**和**选择**是必不可少的控制结构，python中有多种方式实现。

## 1.10.1迭代

python中提供了`while`和`for`两种策略来实现循环迭代：

### while循环

while结构在开始的时候判断**循环条件**，如果条件为`True`，执行循环里的内容，示例代码：

```python
>>> counter = 1
>>> while counter <= 5:
...     print("Hello, world")
...     counter = counter + 1


Hello, world
Hello, world
Hello, world
Hello, world
Hello, world
```

上述代码执行了五次`print("Hello, world")`，通过counter来控制循环条件。

### for循环

for循环结构经常和python中的集合数据结构配合起来使用，用于遍历集合中的各个元素。例如：

```python
>>> for item in [1,3,6,2,5]:
...    print(item)
...
1
3
6
2
5
```

上述代码中把列表中的元素依次赋予`item`变量，然后通过print函数显示。

下面是一个for循环来获取单词中字符的例子：

```python
wordlist = ['cat','dog','rabbit']
letterlist = [ ]
for aword in wordlist:
    for aletter in aword:
        letterlist.append(aletter)
print(letterlist)
```

## 1.10.2 判断选择

大多数编程语言提供两种选择策略：`ifelse`和`if`。用`ifelse`实现一个简单的二分支的选择：

```python
if n<0:
   print("Sorry, value is negative")
else:
   print(math.sqrt(n))
```

上述 代码中是对一个非负数n求平方根，如果n是小于0的话用print输出出错信息。

此外，判断选择模块也可以进行嵌套，比如下面示例是根据计算机科学考试的得分`score`判断该门课程的等级：

```python
if score >= 90:
   print('A')
else:
   if score >=80:
      print('B')
   else:
      if score >= 70:
         print('C')
      else:
         if score >= 60:
            print('D')
         else:
            print('F')
```

另一种实现方式是用`elif`关键字来实现多分支的选择：

```python
if score >= 90:
   print('A')
elif score >=80:
   print('B')
elif score >= 70:
   print('C')
elif score >= 60:
   print('D')
else:
   print('F')
```

而`if`策略，则只关注满足条件的情况，省略对`else`部分的处理：

```python
if n<0:
   n = abs(n)
print(math.sqrt(n))
```



## 1.10.3 **list comprehension**

用循环迭代和选择的方式构建**列表**被称为**列表推导式(list comprehension)**：

下面示例是计算1-10的平方，并构建列表：

```python
>>> sqlist=[]
>>> for x in range(1,11):
         sqlist.append(x*x)

>>> sqlist
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>>
```

上面示例中采用for循环，在循环体里面计算平方，将结果加入结果列表`sqlist`。

用列表推导式实现方式如下：

```python
>>> sqlist=[x*x for x in range(1,11)]
>>> sqlist
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
>>>
```

变量`x`从range(1,11)中依次获取元素，`x*x`为计算结果，存入`sqlist`结果中。

下面示例是用列表推导式返回1-10中奇数的平方列表：

```python
>>> sqlist=[x*x for x in range(1,11) if x%2 != 0]
>>> sqlist
[1, 9, 25, 49, 81]
```

在列表推导式中加入`if x%2!=0`的条件来筛选奇数。

下面是通过列表推导式把字母列表中的非元音字母提取出来并转换为大写，形成列表：

```python
>>>[ch.upper() for ch in 'comprehension' if ch not in 'aeiou']
['C', 'M', 'P', 'R', 'H', 'N', 'S', 'N']
```

