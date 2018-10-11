python数据结构，1.12节函数(https://runestone.academy/runestone/static/pythonds/Introduction/DefiningFunctions.html)

## 定义函数

函数定义由定义符号`def`、函数名、参数(在括号里)、函数体(可能包含返回参数)组成：

比如下面示例：

```python
>>> def square(n):
...    return n**2
...
>>> square(3)
9
>>> square(square(3))
81
```

示例定义了一个求平方的函数，函数名为`square`，参数为`n`，函数体为：`return n**2`，这个函数体将函数操作和返回参数结合在一起了，可以拆分为：

```python
def square(n):
	result = n ** 2
	return result
```

其中`return`关键字后面加的就是返回参数。

## 牛顿法求平方根

```python
def squareroot(n):
    root = n/2    #initial guess will be 1/2 of n
    for k in range(20):
        root = (1/2)*(root + (n / root))
    return root
```

运行结果：

```
>>>squareroot(9)
3.0
>>>squareroot(4563)
67.549981495186216
```

