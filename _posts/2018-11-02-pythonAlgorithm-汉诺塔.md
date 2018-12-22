# python算法——汉诺塔问题

汉诺塔问题是递归算法解决的经典问题之一：

* 初始设定：有三根杆A，B和C；A杆上套着N个大小不同的盘子，下面盘子比上面盘子大
* 要求：将所有盘子从A杆移到C杆
* 条件：每次仅能一个盘子，在此过程中保证每个杆上下面盘子比上面盘子大

该问题满足递归算法解决问题的三个条件，可以用递归解决：

* 递归终止条件：A中所有盘子被移到C中
* 每一次递归：将A中最底层的盘子向C移动，A中盘子数减1
* 算法调用本身：A中最下面盘子的上部看作是另一个相同的问题，调用算法本身解决

代码示例如下：

```python
def moveTower(height,fromPole, toPole, withPole):
    if height >= 1:
        moveTower(height-1,fromPole,withPole,toPole)
        moveDisk(fromPole,toPole)
        moveTower(height-1,withPole,toPole,fromPole)

def moveDisk(fp,tp):
    print("moving disk from",fp,"to",tp)

moveTower(3,"A","B","C")

```

