python数据结构——[队列](https://runestone.academy/runestone/static/pythonds/BasicDS/ImplementingaQueueinPython.html)

# 3.12队列的实现

队列是一种先入先出（First In First Out的数据结构，python中队列实现如下：

```python
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
```

上述代码中实现了队列的初始化`__init__`，队空`isEmpty`，入队`enqueue`和出队`dequeue`方法，使用示例如下：

```python
q=Queue()	
q.enqueue(4)
q.enqueue('dog')
q.enqueue(True)
print(q.size())
```

# 3.13 队列应用

模拟“Hot potato”问题，一个类似于击鼓传花的问题，最后淘汰到只剩一个人。

```
from pythonds.basic.queue import Queue
from random import randint

def hotPotato(namelist, num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())

        simqueue.dequeue()

    return simqueue.dequeue()

print(hotPotato(["Bill","David","Susan","Jane","Kent","Brad"],randint(1,100)))
```

这里采用了一个队列来存储所有小朋友，hot potato在队列首的小朋友手里，每一轮队首的小朋友出队（相当于把土豆交给下一个小朋友），然后再入队（插入队尾）；当持续num轮后，队首的小朋友被淘汰（手中有热土豆）；最终队列中只剩一个小朋友时游戏结束。

这里稍微做了点改动，引入了random模块产生随机数控制num轮数。

# 3.14 打印任务模拟

假设在一个实验室中平均每天在任一小时有10个学生都在工作，每个学生每小时最多会打印两次，每次的打印任务时1-20页。打印机是一台老旧打印机，以较差质量打印速度为10页/分钟，以较好质量打印速度为5页/分钟。不同打印质量需要的打印时间到底相差多少？可以用一个小程序模拟这一过程。

我们用一个队列来存储打印任务，当打印任务完成则从队列中移除，这样上述问题就等价于一个求队列中任务平均等待时间的问题。在假设模型中需要用到一些概率假设：每篇文档长度为1-20页的概率是相等的。假设每个小时内10个学生没人都会打印2篇文档，总共会产生20个打印任务，那么每一秒产生打印任务的概率是1/180，我们可以通过每秒产生一个1-180之间的随机数，如果刚好是180的话表示产生打印任务（其实也可以任一一个数，前提是随机数产生是均一分布的）。

```python
from pythonds.basic.queue import Queue

import random

class Printer:
    def __init__(self, ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = 0

    def tick(self):
        if self.currentTask != None:
            self.timeRemaining = self.timeRemaining - 1
            if self.timeRemaining <= 0:
                self.currentTask = None

    def busy(self):
        if self.currentTask != None:
            return True
        else:
            return False

    def startNext(self,newtask):
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages() * 60/self.pagerate

class Task:
    def __init__(self,time):
        self.timestamp = time
        self.pages = random.randrange(1,21)

    def getStamp(self):
        return self.timestamp

    def getPages(self):
        return self.pages

    def waitTime(self, currenttime):
        return currenttime - self.timestamp


def simulation(numSeconds, pagesPerMinute):

    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []

    for currentSecond in range(numSeconds):

      if newPrintTask():
         task = Task(currentSecond)
         printQueue.enqueue(task)

      if (not labprinter.busy()) and (not printQueue.isEmpty()):
        nexttask = printQueue.dequeue()
        waitingtimes.append( nexttask.waitTime(currentSecond))
        labprinter.startNext(nexttask)

      labprinter.tick()

    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining."%(averageWait,printQueue.size()))

def newPrintTask():
    num = random.randrange(1,181)
    if num == 180:
        return True
    else:
        return False

for i in range(10):
    simulation(3600,5)
```

上述代码模拟了一个小时内打印任务的等待时间，其中调整simulation的参数5可以改变打印质量，查看不同质量下等待时间的差别。