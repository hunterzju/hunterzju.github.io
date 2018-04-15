对[runestone.academy](https://runestone.academy/runestone/static/pythonds/Introduction/toctree.html)网站上python数据结构和算法的课程简单总结，前面内容不做总结了，直接从干货开始记录。

# 1.8数据类型

python中的数据类型主要包含**原子数据类型(atomic data types)**和**集合数据类型(collection data types)**。

## 1.8.1原子数据类型

主要包含了**整型(int)**，**浮点型(float)**和**布尔型(bool)**。

对于**整型**和**浮点型**的数据可以有这么几种操作：

加(+)，减(-)，乘(*)，除(/)，乘方(**)，取余(%)和整除(//)

示例如下：

```python
print(2+3*4)
print((2+3)*4)
print(2**10)
print(6/3)
print(7/3)
print(7//3)
print(7%3)
print(3/6)
print(3//6)
print(3%6)
print(2**100)
```

运行结果：

```python
14
20
1024
2.0
2.33333333333
2
1
0.5
0
3
1267650600228229401496703205376
```

针对**布尔型**数据的操作有**与(and)**，**或(or)**，**非(not)**三种；

**布尔型**数据的取值也只能是**True**和**False**两个，示例和运行结果如下：

```python
>>> True
True
>>> False
False
>>> False or True
True
>>> not (False or True)
False
>>> True and True
True
```

同时，**布尔型**数据还作为一些比较运算符的返回结果，如下表所示：

|  操作名  | 操作符 |                解释                 |
| :------: | :----: | :---------------------------------: |
|   小于   |   <    |   左边小于右边返回True，否则False   |
|   大于   |   >    |   左边大于右边返回True，否则False   |
| 小于等于 |   <=   | 左边小于等于右边返回True，否则False |
| 大于等于 |   >=   | 左边大于等于右边返回True，否则False |
|   等于   |   ==   |     两边相等返回True，否则False     |
|  不等于  |   !=   |     两边不等返回True，否则False     |
|  逻辑与  |  and   |   两边都是True返回True，否则False   |
|  逻辑或  |   or   |       有一边为True就返回True        |
|  逻辑非  |  not   |   把True变成False，False变成True    |

示例如下：

```pyhton
print(5==10)
print(10 > 5)
print((5 >= 1) and (5 <= 10))
```

运行结果如下：

```python
False
True
True
```

***tips:***

**变量**可以用来标识数据：

* 一般变量只能由英文字符(a-zA-Z)，数字(0-9)和下划线(_)组成
* 变量开头必须是下划线或者英文字符
* python中变量会在第一次被赋值时创建
* python中变量中只保存对数据的引用，而不是数据本身
* python中变量可以动态指向不同数据类型(见下面示例)

```python
>>> theSum = 0
>>> theSum
0
>>> theSum = theSum + 1
>>> theSum
1
>>> theSum = True
>>> theSum
True
```

在这个过程中```theSum```变量一开始指向整型，然后指向布尔型，这是python与其他变成语言变量的不同之处。

## 1.8.2集合数据类型

主要包含**列表(list)**，**字符串(strings)**，**元组(tuple)**，**集合(set)**和**字典(dictionary)**。

### 列表

列表是一种可以按照顺序存储多个原子数据的集合数据类型。

* 列表通过"[]"来表示，列表中不同元素用逗号(,)分割开。
* 列表中元素可以是不同的数据类型。

示例如下：

```python
>>> [1,3,True,6.5]
[1, 3, True, 6.5]
>>> myList = [1,3,True,6.5]
>>> myList
[1, 3, True, 6.5]
```

list是一种**序列(sequence)**，python中序列支持的操作如下：

| **操作名** | **操作符** |         **解释**         |
| :--------: | :--------: | :----------------------: |
|  下标索引  |    [ ]     |    访问序列中指定元素    |
|    级联    |     +      |     把序列组合在一起     |
|    重复    |     *      |       重复序列多次       |
|  成员关系  |     in     | 判断一个元素是否在序列中 |
|    长度    |    len     |  判断列表中有多少个元素  |
|    切片    |   [ : ]    |   取出序列中的部分元素   |

下面是简单的示例：

```python
myList = [1,2,3,4]
A = [myList]*3
print(A)
myList[2]=45
print(A)
```

运行结果如下：

```python
[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
[[1, 2, 45, 4], [1, 2, 45, 4], [1, 2, 45, 4]]
```

注意点：

* A是通过变量myList*3 得到，A中有3个myList列表
* 修改myList中的数据，A中单个myList列表中对应位置的数据都会变。

列表作为一种集合数据类型，还支持很多操作方法(method)：

| **方法名** |      **使用示例**      |            **解释**             |
| :--------: | :--------------------: | :-----------------------------: |
|  `append`  |  `alist.append(item)`  |     在列表的末尾添加新元素      |
|  `insert`  | `alist.insert(i,item)` |    在列表指定位置添加新元素     |
|   `pop`    |     `alist.pop()`      |   返回并移除列表中末尾的元素    |
|  `remove`  |  `alist.remove(item)`  |      从列表中删除指定元素       |
|   `sort`   |     `alist.sort()`     |         对列表进行排序          |
| `reverse`  |   `alist.reverse()`    |       对列表顺序进行倒置        |
|   `del`    |     `del alist[i]`     |       删除列表中第i个元素       |
|  `index`   |  `alist.index(item)`   | 返回列表中 `item`元素对应的下标 |

示例如下：

```python
myList = [1024, 3, True, 6.5]
myList.append(False)
print(myList)
myList.insert(2,4.5)
print(myList)
print(myList.pop())
print(myList)
print(myList.pop(1))
print(myList)
myList.pop(2)
print(myList)
myList.sort()
print(myList)
myList.reverse()
print(myList)
print(myList.count(6.5))
print(myList.index(4.5))
myList.remove(6.5)
print(myList)
del myList[0]
print(myList)
```

运行结果：

```python
[1024, 3, True, 6.5, False]
[1024, 3, 4.5, True, 6.5, False]
False
[1024, 3, 4.5, True, 6.5]
3
[1024, 4.5, True, 6.5]
[1024, 4.5, 6.5]
[4.5, 6.5, 1024]
[1024, 6.5, 4.5]
1
2
[1024, 4.5]
[4.5]
```

最简单的数据类型也有**方法**，比如：

```python
>>> (54).__add__(21)
75
```

这里对整型数据54调用了`__add__()`方法。

python中还有一个广泛使用的函数range()和列表关系密切，它可以产生一个数值序列，通过list函数转换成列表可以看到里面的数据。例如：

```python
>>> range(10)
range(0, 10)
>>> list(range(10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> range(5,10)
range(5, 10)
>>> list(range(5,10))
[5, 6, 7, 8, 9]
>>> list(range(5,10,2))
[5, 7, 9]
>>> list(range(10,1,-1))
[10, 9, 8, 7, 6, 5, 4, 3, 2]
```

### 字符串

字符串是连续字符序列的集合，python中用引号""(双引号|单引号)来表示。下面是简单示例：

```python
>>> "David"
'David'
>>> myName = "David"
>>> myName[3]
'i'
>>> myName*2
'DavidDavid'
>>> len(myName)
5
```

字符串也是一种**序列**，之前列表中提到过序列支持的操作也适用。此外，字符串还有一些独有的方法，下表是简要总结：

| **方法名** |        **用法**        |                   **解释**                   |
| :--------: | :--------------------: | :------------------------------------------: |
|  `center`  |  `astring.center(w)`   | 返回一个以`astring`为中心，长度为`w`的字符串 |
|  `count`   | `astring.count(item)`  |          返回字符串中 `item` 的个数          |
|  `ljust`   |   `astring.ljust(w)`   | 从字符串左边开始，返回一个长度为 `w`的字符串 |
|  `lower`   |   `astring.lower()`    |         将字符串中所有字母转化为小写         |
|  `rjust`   |   `astring.rjust(w)`   | 从字符串右边开始，返回一个长度为 `w`的字符串 |
|   `find`   |  `astring.find(item)`  |     返回字符串中第一个 `item`对应的下标      |
|  `split`   | `astring.split(schar)` |            根据 `schar`分割字符串            |

示例如下：

```python
>>> myName
'David'
>>> myName.upper()
'DAVID'
>>> myName.center(10)
'  David   '
>>> myName.find('v')
2
>>> myName.split('v')
['Da', 'id']
```

字符串和列表的不同之处在于，列表是**可变的(mutability)**，字符串是**不可变(immutability)**，比如下面的操作：

```python
>>> myList
[1, 3, True, 6.5]
>>> myList[0]=2**10
>>> myList
[1024, 3, True, 6.5]
>>>
>>> myName
'David'
>>> myName[0]='X'
Traceback (most recent call last):
  File "<pyshell#84>", line 1, in -toplevel-
    myName[0]='X'
TypeError: object doesn't support item assignment
```

### 元组

元组和列表类似，里面数据也可以是不同类型的，不同点在于元组是不可变的。python中元组用括号()表示。元组作为一种序列，也支持python中的序列操作。例如：

```python
>>> myTuple = (2,True,4.96)
>>> myTuple
(2, True, 4.96)
>>> len(myTuple)
3
>>> myTuple[0]
2
>>> myTuple * 3
(2, True, 4.96, 2, True, 4.96, 2, True, 4.96)
>>> myTuple[0:2]
(2, True)
```

但是如果想要修改元组中的数据，就会报错：

```python
>>> myTuple[1]=False
Traceback (most recent call last):
  File "<pyshell#137>", line 1, in -toplevel-
    myTuple[1]=False
TypeError: object doesn't support item assignment
```

### 集合

集合是python中一种无序的集合数据类型。集合中元素也可以是不同的数据类型，一般用大括号{}表示，里面的数据用逗号隔开。注意空的集合用`set()`表示，为了和后面的**字典**区分开。

集合不是一种序列数据类型，因此支持的操作和之前的序列数据类型有所区别：

| **操作名称** |     **操作符**     |                     **解释**                     |
| :----------: | :----------------: | :----------------------------------------------: |
|     成员     |         in         |             判断元素是否是集合的成员             |
|     长度     |        len         | 返回集合中有多少个不同的元素（基数cardinality）  |
|     `|`      | `aset | otherset`  |                返回两个集合的并集                |
|     `&`      | `aset & otherset`  |                返回两个集合的交集                |
|     `-`      | `aset - otherset`  | 将`aset`中元素减去`otherset`中元素作为新集合返回 |
|     `<=`     | `aset <= otherset` |       判断`otherset`中元素是否都在`aset`中       |

下面是简单示例：

```python
>>> mySet
{False, 4.5, 3, 6, 'cat'}
>>> len(mySet)
5
>>> False in mySet
True
>>> "dog" in mySet
False
```

集合同时还支持一些数学方法，下面是简单总结：

|   **方法名**   |           **用法**            |             **解释**             |
| :------------: | :---------------------------: | :------------------------------: |
|    `union`     |    `aset.union(otherset)`     |        返回两个集合的并集        |
| `intersection` | `aset.intersection(otherset)` |        返回两个集合的交集        |
|  `difference`  |  `aset.difference(otherset)`  |   返回`aset`对`otherset`的补集   |
|   `issubset`   |   `aset.issubset(otherset)`   | 判断`aset`是否是`otherset`的子集 |
|     `add`      |       `aset.add(item)`        |         给set中添加元素          |
|    `remove`    |      `aset.remove(item)`      |         删除set中的元素          |
|     `pop`      |         `aset.pop()`          |    从set中返回并移除任意元素     |
|    `clear`     |        `aset.clear()`         |             清空集合             |

下面是示例：

```python
>>> mySet
{False, 4.5, 3, 6, 'cat'}
>>> yourSet = {99,3,100}
>>> mySet.union(yourSet)
{False, 4.5, 3, 100, 6, 'cat', 99}
>>> mySet | yourSet
{False, 4.5, 3, 100, 6, 'cat', 99}
>>> mySet.intersection(yourSet)
{3}
>>> mySet & yourSet
{3}
>>> mySet.difference(yourSet)
{False, 4.5, 6, 'cat'}
>>> mySet - yourSet
{False, 4.5, 6, 'cat'}
>>> {3,100}.issubset(yourSet)
True
>>> {3,100}<=yourSet
True
>>> mySet.add("house")
>>> mySet
{False, 4.5, 3, 6, 'house', 'cat'}
>>> mySet.remove(4.5)
>>> mySet
{False, 3, 6, 'house', 'cat'}
>>> mySet.pop()
False
>>> mySet
{3, 6, 'house', 'cat'}
>>> mySet.clear()
>>> mySet
set()
```

### 字典 

字典是关系数据对的集合，数据对关系是“键-值(key-value)“型，写作key:value。字典用大括号表示。下面是示例：

```python
>>> capitals = {'Iowa':'DesMoines','Wisconsin':'Madison'}
>>> capitals
{'Wisconsin': 'Madison', 'Iowa': 'DesMoines'}
```

对字典的操作可以通过键key来访问值value，也可以添加键值对，比如：

```python
capitals = {'Iowa':'DesMoines','Wisconsin':'Madison'}
print(capitals['Iowa'])
capitals['Utah']='SaltLakeCity'
print(capitals)
capitals['California']='Sacramento'
print(len(capitals))
for k in capitals:
   print(capitals[k]," is the capital of ", k)
```

运行结果为：

```python
DesMoines
{'Iowa': 'DesMoines', 'Wisconsin': 'Madison', 'Utah': 'SaltLakeCity'}
4
DesMoines  is the capital of  Iowa
Madison  is the capital of  Wisconsin
SaltLakeCity  is the capital of  Utah
Sacramento  is the capital of  California
```

需要注意字典的存储并不是按照顺序存储的。

字典同样支持操作符和方法，下表分别是操作和方法的总结：

**操作**

| **操作符** |     **用法**     |                **解释**                 |
| :--------: | :--------------: | :-------------------------------------: |
|    `[]`    |   `myDict[k]`    | 通过key来获取value，如果key不存在则报错 |
|    `in`    |  `key in adict`  |           判断key是否在字典中           |
|   `del`    | del `adict[key]` |           从字典中移除键值对            |

如下是简单示例：

```python
>>> phoneext={'david':1410,'brad':1137}
>>> phoneext
{'brad': 1137, 'david': 1410}
>>> phoneext.keys()
dict_keys(['brad', 'david'])
>>> list(phoneext.keys())
['brad', 'david']
>>> phoneext.values()
dict_values([1137, 1410])
>>> list(phoneext.values())
[1137, 1410]
>>> phoneext.items()
dict_items([('brad', 1137), ('david', 1410)])
>>> list(phoneext.items())
[('brad', 1137), ('david', 1410)]
>>> phoneext.get("kent")
>>> phoneext.get("kent","NO ENTRY")
'NO ENTRY'
```

**方法**

| **方法名** |      **用法**      |               **解释**               |
| :--------: | :----------------: | :----------------------------------: |
|   `keys`   |   `adict.keys()`   |         返回字典中所有键key          |
|  `values`  |  `adict.values()`  |        返回字典中所有值value         |
|  `items`   |  `adict.items()`   |          返回字典中的键值对          |
|   `get`    |   `adict.get(k)`   | 通过key获取value，key不存在返回None  |
|   `get`    | `adict.get(k,alt)` | 通过key获取value，key不存在返回`alt` |

