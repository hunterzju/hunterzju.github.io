python提供了一些函数用于用户和程序的交互

## 输入

用户可以通过`input()`函数向程序输入信息，使用如下：

```python
aName = input('Please enter your name: ')
```

上述代码提示用户输入用户名，并将输入信息保存到`aName`变量中，读取到的数据是`string`类型，因此可以使用`string`的方法，比如：

```python
aName = input("Please enter your name ")
print("Your name in all capitals is",aName.upper(),
      "and has length", len(aName))
```

在`print()`函数中对`aName`使用了字符串的`upper()`和`len()`方法，将字符转化为大写并计算字符串长度。运行结果如下：

```python
Your name in all capitals is TEST and has length 4
```

如果需要将输入内容当作数字做处理的话，需要进行强制的类型转换：

```python
sradius = input("Please enter the radius of the circle ")
radius = float(sradius)
diameter = 2 * radius
```

这里输入半径r，可以求出直径d，需要先把string类型的输入用`float()`函数转换为float类型。

## 输出

用`print()`函数可以把信息打印到屏幕上，在输出的时候有很多选项可以让输出成为我们想要的形式。比如可以通过`sep`参数指定分隔符，通过`end`参数指定结束符：

```python
>>> print("Hello")
Hello
>>> print("Hello","World")
Hello World
>>> print("Hello","World", sep="***")
Hello***World
>>> print("Hello","World", end="***")
Hello World***
```

有时候需要把变量组合在字符串中输出，比如想要输出：aName, "is", age, "years old."

里面的aName和age是两个变量，剩下的“is”和“years old”是固定不变的字符串。这个时候需要用到**格式化(format)**输出：

```python
print("%s is %d years old." % (aName, age))
```

其中`%`是格式化输出符号，后面的`s`和`d`代表不同的数据类型，对于数据类型的总结如下：

| **字符** | **输出**                                                     |
| -------- | ------------------------------------------------------------ |
| `d`, `i` | 整数                                                         |
| `u`      | 无符号整数                                                   |
| `f`      | 浮点数，格式为 m.ddddd                                       |
| `e`      | 浮点数，格式为 m.ddddde+/-xx                                 |
| `E`      | 浮点数，格式为 m.dddddE+/-xx                                 |
| `g`      | 用 `%e` 表示 小于10^ (-4) 或者大于10^(+5)的数，否则用`%f`表示 |
| `c`      | 单个字符                                                     |
| `s`      | 字符串                                                       |
| `%`      | 插入占位符                                                   |

此外还可以通过在`%`后面加数字指定格式化的长度：

| **符号** | **示例**   | **说明**                                   |
| -------- | ---------- | ------------------------------------------ |
| number   | `%20d`     | 将变量值转化为长度为20的格式               |
| `-`      | `%-20d`    | 将变量值转化为长度为20的格式，左对齐       |
| `+`      | `%+20d`    | 将变量值转化为长度为20的格式，右对齐       |
| `0`      | `%020d`    | 将变量值转化为长度为20的格式，前面用0补齐  |
| `.`      | `%20.2f`   | 将变量值转化为长度为20的格式，包括两位小数 |
| `(name)` | `%(name)d` | 从字典中跟去键`name`获取值。               |

具体示例如下：

```
>>> price = 24
>>> item = "banana"
>>> print("The %s costs %d cents"%(item,price))
The banana costs 24 cents
>>> print("The %+10s costs %5.2f cents"%(item,price))
The     banana costs 24.00 cents
>>> print("The %+10s costs %10.2f cents"%(item,price))
The     banana costs      24.00 cents
>>> itemdict = {"item":"banana","cost":24}
>>> print("The %(item)s costs %(cost)7.1f cents"%itemdict)
The banana costs    24.0 cents
```

