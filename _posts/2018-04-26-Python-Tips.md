python使用中遇到的一些问题记录：

* lambda表达式

  匿名函数，使用方法:

  ```python
  lambda x:x+1
  ```

  x为输入，：后面为输出

  可以先定义一个匿名函数，然后使用：

  ```python
  a = range(-5,5)
  func = lambda x: x if x>=0 else -1
  a_list = [func(i) for i in a]
  ```

  lambda匿名函数一般用来实现一些简单的函数操作。

* ​