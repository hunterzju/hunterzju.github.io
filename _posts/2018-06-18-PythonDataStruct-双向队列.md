python基础数据结构，[双向队列](https://runestone.academy/runestone/static/pythonds/BasicDS/ImplementingaDequeinPython.html)

双向队列是在队列基础上增加了新的操作，使得队列中元素可以从队列头入队和出队。

# 3.17双向队列在python中的实现

python中对队列的实现也在队列基础上加入了从队列头部入队和出队的操作addFront和removeFront：

```python
class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)
```

下面是双向队列的简单使用示例：

```python
d=Deque()
print(d.isEmpty())
d.addRear(4)
d.addRear('dog')
d.addFront('cat')
d.addFront(True)
print(d.size())
print(d.isEmpty())
d.addRear(8.4)
print(d.removeRear())
print(d.removeFront())
```

输出结果如下：

```python
True
4
False
8.4
True
```

上面代码中首先生成了双向队列的实例d，新生成的实例是空的队列，因而d.isEmpty返回True被打印出来；

接下来操作中，d从队尾加入了4，从队尾加入了'dog'，从队头加入了'cat'，从队头加入了True，此时队列是这样的：

队头 -> True -> cat -> 4 -> dog ->队尾

d.size()返回4；d.isEmpty返回False；

然后d.addRear(8.4)在队尾加入了8.4：

队头 -> True -> cat -> 4 -> dog -> 8.4 ->队尾

此时d.removeRear返回8.4，d.removeFront返回True。

# 3.18 回文序列检测

回文序列就是序列从正向和反向看结果都是一样的，下面是用python进行回文序列检测的简单实例：

```python
from pythonds.basic.deque import Deque

def palchecker(aString):
    chardeque = Deque()

    for ch in aString:
        chardeque.addRear(ch)

    stillEqual = True

    while chardeque.size() > 1 and stillEqual:
        first = chardeque.removeFront()
        last = chardeque.removeRear()
        if first != last:
            stillEqual = False

    return stillEqual

print(palchecker("lsdkjfskf"))
print(palchecker("radar"))
```



