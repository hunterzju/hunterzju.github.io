python数据结构和算法[1.10节](https://runestone.academy/runestone/static/pythonds/Introduction/ExceptionHandling.html)：

## 1.11 异常处理

在编程的时候通常会出现两类错误。

* 语法错误

  第一类是语法错误，就是没有按照程序语言的语法规则来编写，比如：

  ```python
  >>> for i in range(10)
  SyntaxError: invalid syntax (<pyshell#61>, line 1)
  ```

  上述语句在for循环后面忘记加冒号(:)了。python的解释器(interpreter)会发现无法处理这条语句，因为没有按照规则来写。语法错误是初学者常犯的错误。

* 逻辑错误

  另外一类是逻辑错误，是指程序可以执行，但是会返回错误的结果，这可能是底层算法的错误或者程序员自己实现算法的时候出错导致的。有时候这类错误会产生灾难性的影响，比如除数为0的情况，或者数组的访问越界。逻辑错误会导致程序在运行过程中出错并停止，这类运行时出现的错误也被叫做**异常**。

  大多数编程语言都提供了一种方法，允许程序员在异常出现的时候，根据异常情况做一些处理，程序员也可以自己定义一些异常情况。

  当异常出现时，计算机术语叫做异常被抛出("raised")，程序员可以用`try`语句来处理(handle)抛出的异常。比如说下面示例：

  ```python
  >>> anumber = int(input("Please enter an integer "))
  Please enter an integer -23
  >>> print(math.sqrt(anumber))
  Traceback (most recent call last):
    File "<pyshell#102>", line 1, in <module>
      print(math.sqrt(anumber))
  ValueError: math domain error
  ```

  我们希望用户输入一个整数作为`anumber`，然后求这个数的平方根，如果这个数大于等于0可以正确求解，但是如果小于0，求平方根就会造成一个`ValueError exception `。

  我们可以通过如下方式用`try-except`语句来处理异常：

  ```python
  >>> try:
         print(math.sqrt(anumber))
      except:
         print("Bad Value for square root")
         print("Using absolute value instead")
         print(math.sqrt(abs(anumber)))
  
  Bad Value for square root
  Using absolute value instead
  4.79583152331
  ```

  这里如果是非负数的话，就正常执行try下面的语句，如果是负数的话，就会触发except下面的语句，对负数求一个绝对值再求平方根。

  程序员也可以用`raise`语句来抛出一个运行时的异常情况，让程序停止，比如下面示例：

  ```python
  >>> if anumber < 0:
  ...    raise RuntimeError("You can't use a negative number")
  ... else:
  ...    print(math.sqrt(anumber))
  ...
  # 运行结果
  Traceback (most recent call last):
    File "<stdin>", line 2, in <module>
  RuntimeError: You can't use a negative number
  ```

  这里用if语句判断，如果anumber小于0的话，就通过`raise`语句抛出异常("RuntimeError")。

  ## 总结：

  **语法错误**在程序运行前就会被检查出来，**逻辑错误**在程序运行时被发现。

  逻辑错误：

  * 可以通过`try-except`语句进行 处理
  * 可以通过`raise`语句人工抛出异常

  

  

